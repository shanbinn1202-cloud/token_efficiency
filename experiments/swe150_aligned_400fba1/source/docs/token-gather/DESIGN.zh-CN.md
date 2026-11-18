# Token 聚集：在选中页之外保留单个历史 token

[English](DESIGN.md) | 简体中文

分支 token-gather-1114，基于 ledger-int-1112（`110632b`）；2026-11-14 的设计。默认关闭；不带 `--token-gather` 的服务器不变（其选择契约、回执、计划和证据逐字节不变；`tools/policy_parity.py` 对比 `110632b`，见第 8 节）。目标：Qwen3.6-27B、thinking 开、8,192 token 预算；Gemma 4 被拒绝。它的 GPU 运行（作业 3602601、3603240、3603443 和测速作业 3611486）及其判定见 [docs/status.zh-CN.md](../status.zh-CN.md)；作业 3602601 之后的布局修复见第 6 节。

## 1. 做什么

Selector V2 按整个 16-token 页选择。goal_query 作者的研究还保留单个 token：条件 E 保留四个协议标记的每一次历史出现，条件 C 保留最新的 1,024 个分隔符 token。带 `--token-gather` 时，`goal_query` 或 `observation_query` 服务器在页选择之外保留这类 token：由一个 *token 挑选器* 指定它们；restore 时把它们的 K/V 行复制（*聚集*）到新的 16-token 全注意力块中，请求的行在选中页之后、run 之前读取这些块。

```sh
python -B serve.py --release-dir "$RELEASE" --task task.json --output "$OUT" --policy observation_query \
  --page-tokens 16 --active-token-budget 8192 --enable-thinking --token-gather markers
```

| 选项 | 含义 |
| --- | --- |
| `--token-gather markers` | 挑选器 `markers_v1`：在 mandatory 与 recent 之外的候选页中，每个标记的每一次出现，最新优先 |
| `--token-gather-marker TEXT` | 一个标记（可重复；默认 `</think>`、`<\|im_end\|>` 和 `<\|im_start\|>user`）。标记是一个 token 序列的文本，用会话的分词器编码；每次出现保留它的第一个 token，所以 `<\|im_start\|>user` 保留每条 user 消息的 `<\|im_start\|>`（在 `mswea_xml_v1` 中即观察；该协议没有 `<tool_response>`） |
| `--token-gather separators` | 挑选器 `separators_v1`：这些页中最新的分隔符 token（解码文本非空且只含空格、制表符、换行和 `. , ? ! ; :`） |
| `--token-gather-quota N` | 每轮最多 N 个挑选，最新优先（markers 默认 0，即不设上限；separators 默认 1,024） |

范围：`goal_query` 和 `observation_query`（Goal 或观察的余弦顺序），`--protocol mswea-xml`，`--page-tokens 16`（解耦布局）。启动器和启动配置拒绝其他所有策略（包括查询插件）、协议和布局，以及 Gemma 4 配置和 `--release-idle-pages`（第 6 节）。

## 2. 为什么聚集后的注意力是精确的

- **K 在 RoPE 之后存储。** `qwen3_next.py` 的 `_project_qkv_gate` 返回归一化、RoPE 之后的 q、k、v，`forward` 把 k 写入分页缓存（Qwen3.5 和 3.6 复用这个注意力类）。因此被聚集的 key 保留其逻辑位置的 RoPE 相位。query 在自己的逻辑位置旋转（`positions.py`：紧凑槽位从不替代 RoPE 位置）。所以 q·k 只依赖逻辑距离；部分旋转（0.25）和交错 MRoPE 分段不改变这一点。
- **prefix 的顺序无关紧要。** FlashAttention 通过块表读取请求的 key，直到物理序列长度；因果掩码只区分 prefix（run 之前的每个块）与 run，而 softmax 注意力与 key 的顺序无关。聚集块放在 prefix 的任何位置，输出都相同。
- **物理位置**只影响全注意力的 slot mapping 和序列长度：行为 `[选中页][聚集块][run]`，`SparsePlan(L, L − |A_t|·P − G)`，G 为聚集的 token 数。
- **48 个 Gated-DeltaNet 层**保持其整段历史的递归状态（carry）。任何选择（页或 token）都不触及它们。
- **不屏蔽任何槽位。** 带块表的 FlashAttention 会注意 prefix 块的每一个槽位。空槽、零槽或重复槽都会改变 softmax（零 key 会给分母加上 exp(0)），因此聚集数量用真实历史 token 补齐到整块：即 *pad*，取没有被其他部分注意到的最新候选 token（`newest_unattended_v1`）。回执把 pad 与挑选一起记录；聚集块的每个槽位都是一个被记录的 token。run 之前也不能有不满的块：run 的块在 phase 1 中按其逻辑位置决定的偏移写入。

CPU 测试在一个玩具 RoPE 注意力层上验证这一点，其缓存由运行时自己的聚集代码填充（`tests/test_token_gather.py`，`GatheredAttentionTests`）：聚集后的行与只在被记录 token（选中页、挑选、pad、run）上的稠密注意力在 1e-12 内相等；移动 prefix 块不改变结果；零 pad 槽或按物理槽位旋转的 key 会改变输出。

## 3. 选择、回执与证据

**预算（`shared_blocks_v1`）。** 聚集块与页按块数共享填充空间：room = capacity − mandatory − recent。挑选截到 room × 16 个 token（优先级最低的先丢）。填充沿 Selector V2 的余弦顺序前进：当这一页加上已选页之外的挑选的聚集仍然放得下时选这一页：`已选 + 1 + ceil(剩余挑选 / 16) ≤ room`。落在已选页内的挑选不再额外占用；没有挑选时填充就是 Selector V2 自己的填充。mandatory 和 recent 页不变，其中的 token 也从不被挑选。

**契约。** 只有这样的服务器的选择契约才多出键 `token_gather`：`{picker, markers, quota, pad: newest_unattended_v1, budget: shared_blocks_v1}`。其摘要（`contract_sha256`）覆盖该钉；其他契约字节和摘要都不变。

**回执与计划。** 有聚集的一轮，签名回执的 body 多出 `gather: {tokens, pad}`（聚集的 token 升序，含挑选与 pad；pad 是其子集）；`ranking` 和 `selected_count` 仍是页选择。原生计划多出 `gather: {tokens, blocks}`，其 `sparse` 的 dropped 为 L − |A_t|·P − G。适配器从已认证的回执取聚集；worker 检查它（整块、低于 L、在选中页之外）。因此原生文件两者都有。

**证据。** 每个激活轮在 `runtime.jsonl` 和回执中的 `query.selection.gather`：`picker`、`picks`（经过上限和 room 之后）、`truncated`（因 room 被丢的挑选）、`covered`（落在已选页中的挑选）、`gathered`、`pad`、`blocks`、`plain_retrieved`（同一轮的普通余弦填充）、`retrieved`、`displaced`（普通填充中未被选的页）、`added`（普通填充之外被选的页）。`selected_pages`、`retrieved_pages` 和 `added/retained/removed_pages` 为最终结果。

## 4. Restore 与 phase 1

**Phase 2。** 引擎在 restore 目标之后分配聚集块（每个全注意力组每 16 个 token 一块）；和 restore 目标一样，它们进入清零队列，并按 `[A_t][聚集块][run]` 装入行。在第一个 phase-2 step，worker 与 restore 目标和临时块一起清零它们，restore 各页，然后聚集每一个槽位（`TensorIO.gather_tokens`）：GPU 上某页（驻留页，或 phase 1 补满的 tail 页）中的 token 在设备上读取；其他 token 从其 DRAM 页（arena 页或 phase-1 的 tail 捕获）在主机上聚集到一个缓冲区，每个全注意力张量复制一次；每个张量一次 `index_put_` 写入所有槽位，最后一次 join。worker 封存 run 时跳过聚集块。

**轮次之间。** 提交时会话 pin 住本轮的聚集块（引擎 `PrivateBlockOwner.gathers`、worker 的会话镜像），并释放上一轮的。下一次预留重新聚集到新块中（之前的块在它提交时释放）。因此会话一直持有其最后一轮的聚集块，直到关闭（那一轮没有聚集则没有）。在 session 池上，关闭回执（`POST /v1/sessions/close` 的响应及其 `sessions.jsonl` 行）把它们单列为 `released.gather_blocks`，与 `resident_pages` 和 `tail` 并列（gather-receipt-1115），所以 `released.blocks` = `resident_pages` + `tail` + `gather_blocks`（Qwen3.6-27B 上只有一个全注意力组）；不带 `--token-gather` 时 `gather_blocks` 为 0，其他字段和 schema（`remote-ra-session-event-v1`）不变。

**Phase 1。** Phase 1 读取 `[R][聚集块][tail][fresh]`，`SparsePlan(Q, Q − |R|·P − G)`：它恰好注意上一轮 phase 2 所注意的内容加上那一轮封存的页，因此观察向量在与以前相同的上下文上计算。只有这时 phase-1 租约才多出 `gathered: {tokens, block_ids}`；聚集块是借用的（租约从不清零或释放它们）。

**容量。** 准入门把聚集块算作预算内的新 phase-2 块：restore + 聚集 ≤ min(预算, 未驻留的已封页 + 上限对应的块数)。

## 5. 代价

**模型。** 一个被聚集的 token 在 16 个全注意力层中各搬运其 K 和 V：16 × 2 × 4 个 KV 头 × 256 × 2 字节 = 64 KiB（一个 16-token 页为 1 MiB）。来自 DRAM 时，每个层张量一次主机聚集加一次 H2D；来自驻留页时是一次设备索引。页 restore 不论页里有什么都搬运 1 MiB，所以聚集 k 个不同页中的 k 个 token 搬运 k × 64 KiB，而 restore 这些页要搬运 k MiB。聚集块只占 HBM：每轮预算中的 G/16 块。

**CPU 代理测量**（单线程、非锁页 arena、16 层中的 4 层；是代码路径的代理，不是 GPU 数字）：来自 16 个不同 DRAM 页的 16 个 token，聚集 0.53 ms，restore 这 16 页 1.23 ms；128 个页的 128 个 token 为 2.7 ms 对 10.9 ms；每页紧排 16 个时，CPU 上 restore 更便宜（8 页的 128 个 token：聚集 2.6 ms，restore 0.7 ms），因为 CPU 逐元素索引，而 restore 复制连续片段。GPU 上的测量（复制时间对 restore 时间，按层和按轮）是 GPU 计划的第一项。在按 token 填充的规模下（第 9 节第 1 项；每轮最多 8,192 个 token），`tools/token_gather_bench.py` 在一个 GPU 上、不加载模型，测量今天的聚集路径、对锁页 arena 的设备直读，以及同样字节量的页 restore（它的 GPU 运行见 [docs/status.zh-CN.md](../status.zh-CN.md)）。

**离线回放**（第 7 节）：E 标记平均每轮聚集 4.5 块（obs262t，thinking 开）和 9.9 块（goalvllm，thinking 关），占 8,192 token 预算的 0.9 % 和 1.9 %，从填充中挤出约 4.5 和 10 页。同样标记的页近似（保护每个含标记的页）占预算的 19–23 %，挤出 51–98 页。

## 6. 拒绝的组合

- **Gemma 4。** 其全局层的 RoPE 与 K/V 布局尚未在 0.24 源码上确认，其窗口布局有自己的 restore 路径。以后再做。
- **其他策略与协议。** 挑选器读取 Qwen XML transcript 的 token 流；random、window、rule 策略和查询插件有各自的填充。
- **其他布局。** P 必须为 16：一个聚集块就是一个 kernel 块，FlashAttention 的块表不能容纳不满的块。聚集按 vLLM 0.24.0 FlashAttention 的 KV 布局 `[blocks, 2, 16, heads, head size]`（块轴 0，token 轴 2）索引，两种步长顺序都可以（NHD，或 HND：运行时的 OffloadingConnector 要求 HND，vLLM 在这里用的就是它）；worker 在 attach 时从已加载后端的 `get_kv_cache_shape` 读出块轴和 token 轴并按它们索引，同时拒绝其他布局和步长顺序；`--triton-attn` 被拒绝。（GPU 作业 3602601 之后修复之前，这项检查和 CPU 测试夹具假定的是 `[2, blocks, 16, heads, head size]`、块轴 1；现在夹具用 vLLM 自己的形状、步长顺序和 reshape 代码构造。）
- **`--release-idle-pages`。** 聚集块没有自己的 DRAM 副本：驱逐空闲会话的页时也得丢掉其聚集块，并在重新准入时从 DRAM 页重新聚集，evict/readmit 记录、容量账本和重新准入的 restore 都要随之改变。token 聚集所服务的策略在 Qwen 的 SWE-bench 运行中都不用该选项（它们的 32 任务运行不带它；只有 `full` 带它跑过，作业 3583081，见[已知限制 2.4](../known-limits.zh-CN.md#24-25-gib-中-16-个-session-的-full-需要---release-idle-pages)），所以启动时拒绝这一组合（启动器和启动配置），引擎所有者也拒绝驱逐 pin 着聚集块的会话。

## 7. 已记录运行的离线回放

在两次已记录的 SWE-32 运行（P = 16，8,192 token 预算）的每个选择性请求上运行 `token_gather.turn`（运行时代码）：回执的余弦顺序、capacity 和安全分区，请求已提交的 token 流，以及固定的 Qwen3.6 分词器。只输出计数和比例。每个请求中回放的普通填充都与记录一致。

| 运行（请求数） | 钉 | 挑选（均值 / p90 / 最大） | 已选页覆盖 | 聚集 + pad | 块数（均值 / p90 / 最大） | 占预算（均值 / p90 / 最大） | 挤出页数（均值 / p90） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| obs262t，observation_query，thinking 开（1,198） | E 标记 | 129 / 266 / 453 | 63 | 66 + 6 | 4.5 / 12 / 20 | 0.9 / 2.3 / 3.9 % | 4.6 / 12 |
| | E 标记，上限 256 | 122 / 256 / 256 | 62 | 60 + 6 | 4.1 / 10 / 15 | 0.8 / 2.0 / 2.9 % | 4.2 / 10 |
| | C 分隔符 1,024 | 1,023 / 1,024 / 1,024 | 361 | 663 + 2 | 41.6 / 59 / 63 | 8.1 / 11.5 / 12.3 % | 41.7 / 59 |
| goalvllm，goal_query，thinking 关（1,677） | E 标记 | 180 / 341 / 464 | 28 | 152 + 6 | 9.9 / 20 / 29 | 1.9 / 3.9 / 5.7 % | 9.9 / 20 |
| | E 标记，上限 256 | 161 / 256 / 256 | 27 | 134 + 6 | 8.7 / 15 / 16 | 1.7 / 2.9 / 3.1 % | 8.8 / 15 |
| | C 分隔符 1,024 | 1,023 / 1,024 / 1,024 | 292 | 731 + 2 | 45.9 / 60 / 63 | 9.0 / 11.7 / 12.3 % | 46.0 / 60 |

没有请求超出其 room（`truncated` 处处为 0）。每个请求的 E 挑选数（129 和 180）与按 token 设计说明对同一映射的计数相同。

## 8. 一致性与测试

- `tools/policy_parity.py` 对比 `110632b` 与本分支：所有结果逐字节相同（契约、摘要、拒绝、启动、计划、`test_decouple_e2e` 运行），只有 `serve.py --help` 和一条用法错误例外，差别仅是三个新选项。
- `tests/test_token_gather.py`：CPU 张量上的聚集复制（设备行、跨块的 arena 页、捕获的页；其他内容不动；任何复制之前就拒绝）、注意力参照、挑选器（增量索引对全量扫描）、填充、pad、记录、契约键、启动器与启动配置的拒绝、容量计数。
- `tests/test_token_gather_e2e.py`：在 (16, 784) 的解耦端到端测试框架上，`observation_query` 和 `goal_query`、默认标记、带上限的参数化标记集、分隔符：每次前向注意到的集合（phase 2：选中页、聚集 token、run；phase 1：R、上一轮的聚集 token、run）、聚集块的内容、两侧的 pin、预算、回执、计划、租约和证据、一次重建、一次取消和关闭（所有块都归还）；以及三个会话一起解码、开启稳定配置的三条解码快速路径：单独、加 `--gdn-device-handoff`、加 `--phase1-in-step`，以及 `--async-scheduling`；还有一个 session 池（六个任务，同时最多打开三个），其关闭回执单列聚集块：一个在最后一轮聚集的会话、两个之前聚集过但最后一轮没有聚集的会话、三个从未聚集的会话，以及不开 token 聚集的同一个池（`gather_blocks` 为 0）。`tests/test_ms_pool.py` 在每次关闭时检查 `released.blocks` = `resident_pages` + `tail` + `gather_blocks`。

## 9. 待定问题

1. **按 token 的 Goal 打分。** Selector V2 每页只有一个向量（页末 token 的隐藏状态）。研究用 Goal 分数最高的 token 填充剩余槽位；这里剩余预算按页余弦填充整页。按 token 打分需要按 token 的向量（页向量的 16 倍）或另一种分数。runtime 侧的设计选项及其代价（2026-11-15 审计）见 [PER-TOKEN-OPTIONS.zh-CN.md](PER-TOKEN-OPTIONS.zh-CN.md)。
2. **按 head 选择**（研究早先的说明）无法实现：一个层组的所有 head 共用一张块表。
3. **增量聚集。** 每次预留都聚集到新块；当聚集的 token 重复时保留上一轮的块可以省去复制（大多数标记逐轮保留）。
4. **标记映射。** `mswea_xml_v1` 没有 `<tool_call>` 或 `<tool_response>`；E 被映射为 `</think>`、`<|im_end|>` 和观察的 `<|im_start|>`。这一映射以及分隔符的定义是否与研究一致，由研究作者确认。
