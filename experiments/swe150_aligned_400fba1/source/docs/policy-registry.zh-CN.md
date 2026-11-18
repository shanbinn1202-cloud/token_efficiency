# 选页策略注册表

[English](policy-registry.md) | 简体中文

policy-registry-1112（基于 integrate-1111 `4bf2c40`；各策略已有的 GPU 运行见 [docs/status.zh-CN.md](status.zh-CN.md)）。每个选页策略只在
`source-overlay/policy_registry.py` 中声明一次，运行时的其他部分都从这里读取策略名和属性。新增一个页级策略就是一个类加一次注册（README 的
“如何添加一个选页策略”一节）。八个策略的 CLI 和 HTTP 接口、启动记录、contract 及其摘要、拒绝信息、receipt 以及证据的键和值都与
`4bf2c40` 相同（第 5 节）。

1. [清单](#1-清单)：`4bf2c40` 中运行时在哪些地方写出或特殊处理了策略名
2. [注册表](#2-注册表)：属性、八个策略、API、插件的填充路径
3. [各模块现在读取什么](#3-各模块现在读取什么)
4. [为兼容而保留的内容](#4-为兼容而保留的内容)
5. [等价性证据](#5-等价性证据)
6. [token 级选择（方案 A：标记页，方案 B：token 聚集）](#6-token-级选择方案-a标记页方案-btoken-聚集)
7. [harness 的后续改动（本分支未做）](#7-harness-的后续改动本分支未做)
8. [query 钩子与 ledger_query（ledger-policy-1112）](#8-query-钩子与-ledger_queryledger-policy-1112)
9. [标记页：goal_query_markers 与 observation_query_markers（marker-pages-1114）](#9-标记页goal_query_markers-与-observation_query_markersmarker-pages-1114)

## 1. 清单

`4bf2c40` 中运行时（不含测试和 `harnesses/`）里每一处写出策略名的地方：对 `serve.py`、`tools/` 和 `source-overlay/` 的每个
`.py` 文件做 AST 扫描（等于某个策略名的字符串常量、提到策略名的文本、对策略常量的引用），再逐处阅读。分类：**S** 选页逻辑，
**B** 预算 / 容量，**F** 功能开关（Goal 文本、Goal 引擎、window / AKV / fill 记录、与 thinking 的交互），**V** 参数校验，
**R** 记录 / 指标，**D** 文档（帮助文本、消息）。

| 分类 | 处数 | 文件 |
| --- | ---: | --- |
| V 参数校验 | 27 | serve.py、remote_ra_http/config.py、selection_contract、query_selection、engine_boundary、private_endpoint、controller、chat_text_protocol、native adapter、私有 native endpoint、frontend_registry、registered_add_bridge、live_frontend、native_authority |
| F 功能开关 | 26 | serve.py、remote_ra_http/config.py、evidence、chat_text_route、selection_contract、query_selection、private_endpoint、phase1_coordinator、window_phase1、goal_overlap、Goal 批处理与 vLLM 引擎的检查、terminus2_goal、chat_text_protocol |
| S 选页逻辑 | 16 | query_selection、selection_contract、phase1_coordinator、window_phase1、native select_pages、remote_ra_growing |
| B 预算 / 容量 | 12 | serve.py、remote_ra_http/config.py、selection_contract、query_selection、capacity、window_capacity |
| R 记录 / 指标 | 7 | serve.py、evidence、selection_contract、query_selection、kv_session_protocol |
| D 文档 | 2 + 16 个文件 | serve.py 的帮助文本、tools/step_profile.py 的标签；另有 16 个模块只在注释、docstring 或日志文本中提到策略名 |
| **在线路径合计** | **90 处** | **30 个文件**（在线路径的 28 个模块、vLLM overlay 中 `selective_policy.py` 的副本、`tools/step_profile.py`） |

不在范围内、未改动：只认识 `full`、`random50`、`goal_query` 和 `observation_query`、且不在在线选页路径上的旧开发层
（`policy_runtime/`、`swebench_baseline/`、`dynamic_runtime.py`、`goal_adapter.py`、PoC 策略适配器 `*_vllm_policy_adapter.py`、
`remote_ra_growing/controller.py`、`remote_ra_linux/live_selector.py` 和 `live_factory.py`、`vllm/v1/worker/goal_producer_bridge.py`、
offloading scheduler 的旧排序计划）：共 21 个文件。

各处（文件、位置、分类：`4bf2c40` 中写的是什么 → 现在读取什么）：

| # | 文件 | 位置 | 类 | `4bf2c40` → 现在 |
| ---: | --- | --- | :---: | --- |
| 1 | serve.py | `POLICIES`、`QUERY_POLICIES`、`WINDOW_POLICIES`、`AKV_RULE` | V | 字面量列表 → 保留（第 4 节），等于注册表中内置策略的视图 |
| 2 | serve.py | `main`：`--policy` 的可选值 | V | `POLICIES` → `names()` |
| 3-4 | serve.py | `prepare`、`prepare_gemma4`：已知策略 | V | `in POLICIES` → `known()` |
| 5 | serve.py | `selection_contract`：Selector V2 的入口检查 | V | `QUERY_POLICIES` → `selector_v2` |
| 6 | serve.py | `selection_contract`：`--fill-seed` | V | `== 'random_fill'` → `pin_options` |
| 7 | serve.py | `selection_contract`：random fill、window 和 AKV 的固定值 | B | 三个分支 → `contract_key`、`pin()` |
| 8 | serve.py | `selection_contract`：`akv_rule` 不接受预算选项、没有预算 | B | `== AKV_RULE` → `budget == 'rule'`、`budget_refusal()` |
| 9 | serve.py | `goal_rewriter_use`：给出的 Goal 模型为何未使用 | R | 按名字的四段文本 → `rewriter_unused` |
| 10-13 | serve.py | `--goal-rewriter-batch`、`--goal-rewriter-engine vllm`、`--goal-overlap`（Qwen 与 Gemma） | F | `== 'goal_query'` → `goal_engine` |
| 14 | serve.py | `--thinking-history strip` 拒绝 `recent_tools` 和 `akv_rule` | F | 名字列表 → `reply_roles`（文本由它们的名字拼接） |
| 15-16 | serve.py | `chat_text_goal` 记录（Qwen 与 Gemma） | F | `== 'goal_query'` → `goal_text` |
| 17 | serve.py | `prepare_gemma4`：Goal 的临时空间 G | B | `== 'goal_query'` → `query_vector == 'goal'` |
| 18 | serve.py | `chat_text_tasks` | V | `chat.POLICIES` → `chat.policies()` |
| 19 | serve.py | 帮助文本、Selector V2 选项组的标题 | D | 不变 |
| 20 | remote_ra_http/config.py | `POLICIES`、`QUERY_POLICIES` | V | 保留（第 4 节） |
| 21-22 | remote_ra_http/config.py | `load`：已知策略；Selector V2 contract 的检查 | V | 列表 → `known()`、`selector_v2` |
| 23 | remote_ra_http/config.py | `admission`：`goal_overlap` | F | `goal_engine` |
| 24 | remote_ra_http/config.py | `window_layout`：Goal 的临时空间 | B | `query_vector == 'goal'` |
| 25 | remote_ra_http/evidence.py | `write_ready`：`selection_fraction` 0.5 | R | `== 'random50'` → `budget == 'half'` |
| 26 | remote_ra_http/evidence.py | `complete`：query 行 | R | 六个名字 → `selector_v2` |
| 27 | remote_ra_http/evidence.py | `complete`：`query.selection.fill` / `.window` | F | 名字 → `evidence_key` |
| 28 | remote_ra_http/chat_text_route.py | `frontend`：Goal 输入 | F | `goal_text` |
| 29 | remote_ra_live_v2/selection_contract.py | `RANDOM_FILL` ... `AKV_TOOL_RESPONSES`、`FILL_POLICIES`、`OPTIONAL_KEYS` | V | 保留（第 4 节） |
| 30 | selection_contract.py | `default_contract`：每个固定值一个关键字参数；rule 无预算、阈值 0 | B | 四个关键字 → 遍历 `pinned()` 的 `**pins`、`budget == 'rule'` |
| 31 | selection_contract.py | `validate`：可选键、每个固定值、只能有一个固定值、rule 的阈值 | V | 名字 → `pinned()`、`validate_pin` |
| 32 | selection_contract.py | Goal 引擎的检查（`uses_goal_...`） | F | `== 'goal_query'` → `goal_engine` |
| 33 | selection_contract.py | `for_policy`：缺少或多出固定值 | V | 三个检查 → 每个带固定值的策略的 `pin_required()`、`pin_foreign()` |
| 34 | selection_contract.py | `capacity_pages`：rule 没有预算 | B | `budget == 'rule'` |
| 35 | selection_contract.py | `random_fill_pin`、`window_pin`、`akv_pin` 及其校验 | S | 保留：内置固定值，由注册表中的类调用 |
| 36 | selection_contract.py | `window_order` | S | 保留：内置顺序，由 `select()` 调用 |
| 37 | selection_contract.py | `window_record` | R | 保留（插件另加 `order_id`），由 `record()` 调用 |
| 38 | selection_contract.py | `akv_kept`、`window_classes` | S | 保留，由 `page_classes()` 调用 |
| 39 | remote_ra_live_v2/query_selection.py | `__init__`：服务的策略 | V | 八个名字 → `known()` |
| 40 | query_selection.py | `__init__`：带固定值的策略的默认 contract | B | 三个分支 → `contract_key`、`pin()` |
| 41-42 | query_selection.py | `__init__`：Goal 批处理 / 引擎检查；Goal 文本自检 | F | `goal_engine`；`goal_text` |
| 43 | query_selection.py | `plan`：rule 的容量；random_fill；window；query 策略 | S | 名字 → `budget`、`kind` |
| 44 | query_selection.py | `plan_reference`、`plan_fast`：`query_source` | S | `query_vector` |
| 45-46 | query_selection.py | `window_fill_selector`：配置中的 `fill`；它的顺序 | R、S | 名字 → `order_id`；`fill(recent, candidates)`（策略的 `select`） |
| 47 | query_selection.py | `window_inputs`：AKV 保留的页；recent_tools 的分类 | S | 名字 → `page_classes()` |
| 48-50 | query_selection.py | `plan_window_fast` / `_reference`：顺序；rule 的不变式；记录 | S、B、R | 名字 → `select()`、`budget == 'rule'`、`record()` |
| 51-52 | query_selection.py | `_begin`：Selector V2 检查；chat-text 的 Goal 输入 | V、F | `selector_v2`；`goal_text` |
| 53 | query_selection.py | `_finish`：Goal 缓存与 Goal forward | F | `query_vector == 'goal'` |
| 54-56 | query_selection.py | `plan_turn`：无 query 向量；Goal 向量；Goal 文本 | S、S、F | `FILL_POLICIES`、名字 → `query_vector`、`goal_text` |
| 57 | query_selection.py | `random_fill_selector`、`plan_random_fill_*` | S | 保留：random_fill 自己的（kind `random`） |
| 58-60 | remote_ra_live_v2/capacity.py | `QUERY`；`selected_max`；`_need_by_pool` 中的 Goal 页 | B | 名字 → `budget`、`selector_v2`；`query_vector` |
| 61 | remote_ra_live_v2/engine_boundary.py | `QUERY`：receipt 的检查 | V | `selector_v2` |
| 62-63 | remote_ra_live_v2/private_endpoint.py | `QUERY`：选页 authority；`--goal-overlap` | V、F | `selector_v2`；`goal_engine` |
| 64 | remote_ra_live_v2/controller.py | `POLICIES` | V | `known()` |
| 65-67 | remote_ra_live_v2/phase1_coordinator.py | Goal forward（两处）；observation 的 query token | F、S、F | `query_vector` |
| 68-70 | remote_ra_live_v2/window_phase1.py | 同上三处 | F、S、F | `query_vector` |
| 71 | remote_ra_live_v2/window_capacity.py | `_gate_turn`：Goal 的临时空间 | B | `query_vector == 'goal'` |
| 72-74 | goal_overlap.py、goal_rewriter_batch.py、goal_rewriter_vllm.py | Goal 引擎的检查 | F | `goal_engine` |
| 75 | remote_ra_live_v2/terminus2_goal.py | `check_config` | F | `goal_text` |
| 76-77 | remote_ra_live_v2/chat_text_protocol.py | `POLICIES` 与 `check_config`；`options` 的 Goal 输入 | V、F | `policies()`（第 4 节）；`goal_text` |
| 78 | remote_ra_live_v2/kv_session_protocol.py | `options`：`fill_seed`、`fill_order`、`budget_mode` `rule` | R | 名字 → 各固定值的 `info()` |
| 79-80 | remote_ra_native_growing/adapter.py | `QUERY` 与 `open`；`reserve` / `complete` | V | `known()`；`selector_v2` |
| 81 | remote_ra_native_growing/contract.py | `select_pages`：full、random50、Selector V2 | S | 名字 → `kind`、`budget == 'half'`（`select_native`） |
| 82 | remote_ra_private_v2/lifecycle.py | native endpoint 只接受 full 和 random50 | V | `not selector_v2` |
| 83-87 | frontend_registry.py（2 处）、registered_add_bridge.py、remote_ra_linux/live_frontend.py、native_authority.py | 路由标签、session、启动 | V | 八个名字 → `known()` |
| 88 | remote_ra_growing/contract.py | `select_random50` | S | 保留：random50 的抽样（fraction 1/2 的 `RandomPageSelectionPolicy`），由注册表调用 |
| 89 | remote_ra_growing/selective_policy.py（及其 vLLM overlay 副本） | request-contract v2 路径的 `POLICY_NAMES` | S | 保留（注册表所基于的 PageSelectionPolicy 协议） |
| 90 | tools/step_profile.py | 提到 `goal_query` 的 profile 标签 | D | 不变 |

只有文档提到策略名的模块：`admission_flags`、`admission_overlap`、`composition`、`deferred_completion`、`goal_async`
（另有一段中止文本）、`goal_text`、`kv_layout`、`lifecycle`、`model_profile`、`quoted_literals`、`registered_route`、
`gemma4_renderer`、`engine_assembly`、`registered_add_protocol` 以及 vLLM engine core（`core.py`、`core_client.py`）。

## 2. 注册表

`source-overlay/policy_registry.py` 是一个顶层模块，导入时只用标准库：`serve.py` 在 release 加入 `sys.path` 之前就要读它（能导入时用
release 源码中的，否则用自己的 `source-overlay` 中的）。每个策略是一个 `PagePolicy` 子类；`register(Class())` 注册它。

### 2.1 属性

| 属性 | 取值 | 读取方 |
| --- | --- | --- |
| `name` | lower_snake_case | `--policy`、启动配置、每个检查 |
| `kind` | `full`、`random`、`query`、`window`、`rule` | 选页计划的分派、`select_pages` |
| `selector_v2` | bool：有 selection contract、两阶段 query 轮次、签名的 query receipt | 启动器、配置、引擎、endpoint、adapter、证据 |
| `query_vector` | `None`、`'observation'`、`'goal'` | 阶段 1 的 query token、Goal forward 与缓存、`query_source`、Goal 临时空间 |
| `goal_text` | bool：query 是 Goal 的文本 | goals.jsonl 中的文本、Terminus 2 的 Goal 输入（`chat_text_goal`）、Goal 文本自检 |
| `goal_engine` | bool：运行 Goal rewriter | `--goal-rewriter-model` 被固定或记录为未使用、`--goal-rewriter-batch`、`--goal-rewriter-engine vllm`、`--goal-overlap` |
| `budget` | `none`、`half`、`fixed`、`rule` | `selected_max`、`capacity_pages`、`default_contract`、启动器对预算选项的拒绝、`selection_fraction` |
| `contract_key`、`pin()`、`validate_pin()`、`pin_options` | 策略在 selection contract 中自己的键（即其名字）或 None | `default_contract`、`validate`、`for_policy`、启动器、kv-session info（`info()`） |
| `evidence_key` | `fill`、`window`、`hook`（query 插件，第 8 节）或 None | 计划的证据、receipt 的 `selection`、`runtime.jsonl` 的 `query.selection` |
| `reply_roles` | bool：区分回复中的工具调用、推理和工具响应 | `--thinking-history strip` 的拒绝 |
| `rewriter_unused` | 文本 | startup.json 的 `goal_rewriter.reason` |
| `order_id` | 文本 | selector 配置的 `fill`、记录的 `order`、填充策略的固定值 `{order}` |
| `protocols` | None（所有协议）或所服务的启动协议（ledger-policy-1112） | 启动器和启动配置的拒绝（`serves`、`protocol_refusal`） |
| `hooked` | bool：query 插件，其余弦填充经过 `query_hook`（第 8 节；自 marker-pages-1114 起也可用 Goal 向量，第 9 节） | 计划（`plan_fast`、`plan_reference`）、query authority 的启动检查（`startup`） |

填充策略（kind `window` 和 `rule`）还有 `select(context)`、`page_classes(view)` 和 `record(context, selected, order, filled)`
（第 2.3 节）。

### 2.2 八个内置策略

| 策略 | kind | selector_v2 | query_vector | goal_text | goal_engine | budget | contract_key | evidence_key | reply_roles | order_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | full | 否 | – | 否 | 否 | none | – | – | 否 | – |
| `random50` | random | 否 | – | 否 | 否 | half | – | – | 否 | – |
| `goal_query` | query | 是 | goal | 是 | 是 | fixed | – | – | 否 | – |
| `observation_query` | query | 是 | observation | 否 | 否 | fixed | – | – | 否 | – |
| `random_fill` | random | 是 | – | 否 | 否 | fixed | `random_fill`（`fill_seed`） | `fill` | 否 | `splitmix64_fisher_yates_v1` |
| `recent_fill` | window | 是 | – | 否 | 否 | fixed | `recent_fill` | `window` | 否 | `newest_first_v1` |
| `recent_tools` | window | 是 | – | 否 | 否 | fixed | `recent_tools` | `window` | 是 | `tool_calls_then_tool_responses_then_reasoning_newest_first_v1` |
| `akv_rule` | rule | 是 | – | 否 | 否 | rule | `akv_rule` | `window` | 是 | `replies_and_latest_tool_responses_v1` |

它们按 `4bf2c40` 的顺序注册（`BUILTIN`）。它们的算法留在原处，由各自的类调用：固定值、`window_order`、`window_record`、`akv_kept`
和 `window_classes` 在 `remote_ra_live_v2/selection_contract.py` 中，random50 的抽样在 `remote_ra_growing/contract.py` 中
（`selective_policy.py` 的 `RandomPageSelectionPolicy`），random_fill 的 selector 和计划在 `query_selection.py` 中，query 策略的余弦排序
在上游 Selector V2 中。

### 2.3 填充路径（插件）

插件是一个填充策略：kind `window`（固定预算）或 `rule`（没有预算：它的 `page_classes` 把要保留的页作为第一个集合返回，和
`akv_rule` 一样）。每个生效的轮次先运行 Selector V2 的边界检查、必选页（初始 prompt）和最近安全页，再调用策略：

- `select(context)`，基于 `remote_ra_growing/selective_policy.py` 的 PageSelectionPolicy 协议：输入 `FillContext`，返回
  `PageSelection`，其 `logical_pages` 是填充顺序，即 `context.candidates` 的一个排列（会检查）；服务取其中前
  `capacity - mandatory - len(recent)` 个。context 包含 `num_pages`、`page_tokens`、`mandatory`、`recent`、`candidates`、
  `capacity`、`local_turn`、`turn`（请求序号）、`session_index`、`turn_ends`、`observation_spans`、`stable_end`、
  `page_classes` 的两个页集合、contract 中的 `pin` 以及候选页的向量（`page_vectors`）；
- `page_classes(view)`：供 `select` 和 `record` 使用的两个页集合（默认为空）；
- `record(...)`：策略在 `window` 下的记录（默认：`window_record` 的公共字段加上它的 `order_id`）。

插件的 contract 在它的名字下固定 `{order: order_id}`（缺少时，或出现在其他策略的 contract 中时，`for_policy` 会拒绝）；启动器接受
`--active-ratio` / `--active-token-budget`（kind `window`），拒绝 `--fill-seed`；给出的 `--goal-rewriter-model` 记录为未使用。插件不能声明
query 向量、Goal 文本或引擎、固定值选项或其他证据键（`register` 会拒绝）。

自 ledger-policy-1112 起，query 插件（kind `query`，用 observation 向量的余弦，并在填充上加一个钩子）有了自己的路径（第 8 节）；
自 marker-pages-1114 起，它也可以像 `goal_query` 一样改用 Goal 向量排序（第 9 节）。暂不支持：对页向量做新打分的插件，以及自成
kind `random` 的插件（它需要像 random_fill 那样有自己的 selector 配置）。

### 2.4 注册与 API

- `register(policy)`、`unregister(name)`（仅插件）、`registered(policy)`（一个代码块内，供测试）；`PLUGINS = ('module:Class', ...)`
  和 `load(entries)`：release 中的插件就是 `source-overlay/page_policies/` 下单独模块里的一个类，加上 `PLUGINS` 中的一项；
- `get(name)`（其他任何值返回 None）、`known`、`names`、`policies`、`where(**properties)`、`builtin_where(**properties)`、`pinned()`，
  以及检查用的简写 `selector_v2`、`query_vector`、`goal_text`、`goal_engine`、`reply_roles`；
- `describe()`：把每个策略输出为一行 JSON（名字、是否内置、各属性、默认固定值），供运行时之外的读取方使用（第 7 节）。

## 3. 各模块现在读取什么

即第 1 节表格的“现在”一列。简言之：名字列表变成 `known()`；Selector V2 的集合（`QUERY`、`QUERY_POLICIES`、`FILL_POLICIES`）变成
`selector_v2` 和 `query_vector`；每个 `== 'goal_query'` 按它控制的内容变成 `goal_engine`、`goal_text` 或 `query_vector == 'goal'`；
固定值变成 `pinned()` 以及各策略的 `pin()`、`validate_pin()`、`pin_required()`、`pin_foreign()` 和 `info()`；预算变成 `budget`；记录变成
`evidence_key` 和 `record()`；计划的分派变成 `budget` 和 `kind`。拒绝文本由各策略自己的文本拼成，因此对八个策略逐字节相同。

## 4. 为兼容而保留的内容

- `serve.py` 的 `POLICIES=(...)`、`QUERY_POLICIES`、`WINDOW_POLICIES`、`AKV_RULE`，以及 `remote_ra_http/config.py` 的
  `POLICIES=(...)`、`QUERY_POLICIES`：harness 从封存的源码中读取 `POLICIES=(...)` 这两行（第 7 节）。`chat_text_protocol.POLICIES`：
  chat-text 拒绝信息中的顺序；`policies()` 在其后追加插件。
- `selection_contract` 的 `RANDOM_FILL`、`RECENT_FILL`、`RECENT_TOOLS`、`WINDOW_ORDERS`、`WINDOW_POLICIES`、`AKV_RULE`、
  `AKV_RULE_ID`、`AKV_TOOL_RESPONSES`、`RANDOM_FILL_RNG`、`DEFAULT_FILL_SEED`、`FILL_POLICIES`、`OPTIONAL_KEYS` 及其函数名
  （harness 的标记、测试）。
- `capacity`、`engine_boundary`、`private_endpoint` 和 native adapter 的 `QUERY`、`controller.POLICIES`，以及 `serve.py` 的
  `GOAL_REWRITER_UNUSED*` 文本：现在是注册表对八个策略的视图。

`tests/test_policy_registry.py` 逐一对照注册表检查它们，并用 AST 扫描确认在线路径的模块不在其他地方写策略名。这些列表描述的是
`4bf2c40` 的八个策略；插件只出现在注册表中，因此新增插件从不需要改它们。

## 5. 等价性证据

- **跨源码树探针**（`tools/policy_parity.py`，在 `4bf2c40` 和本分支的 Git 导出上、用同一个工作目录运行）：9,524 个结果，全部逐字节相同
  （两个 JSON 文件的 SHA-256 相同）：P × 向量格式 × 比例 / 预算 × 变体 × Goal 模型 × 每种固定值的 contract，以及每个策略在其上的
  `for_policy`；21 个畸形 contract 与固定值组合上的 `validate` 和 `for_policy`；各固定值；`capacity_pages`；策略 × n × 轮次上的
  `selected_max` 和 `select_pages`；kv-session 记录；每个策略 × 14 组选项 × 3 种布局下启动器的 selection contract 和 Goal 记录；每个策略
  × 7 组选项 × 12 组协议 / 开关（Qwen）以及 × 6 × 5（Gemma 4）的 `serve.prepare`，并对每个启动的配置做 `config.load`、kv-session info、
  chat-text options 和检查；dry-run 的 `main` 和 `--help`；test_decouple_e2e 中 full、random50、goal_query 和 observation_query 在
  P = 784 和 16 上的 stage-B 脚本（语义、分配、RPC 轨迹、runtime 行）。
- **测试**（固定值未改）：test_recent_fill 的 144 个默认 contract 及其 recent_fill / recent_tools 的 contract、receipt 和 stage-B 输出，
  test_random_fill 和 test_akv_rule 的同类固定值，test_random_fill_legacy 与旧代码的等价性（它的 AST 检查现在先还原注册表查询，
  `REGISTRY_LATER`），test_selection_contract、test_selection_fast、test_decouple_selection 以及 release check。test_recent_fill 和
  test_akv_rule 中对字面量列表的检查现在覆盖剩下的列表。
- **harness 的读取函数**：perf-swe32、perf-mab 和 perf-gemma-swe32 的 `runtime_features`（它们在 `4bf2c40` 中的代码）在本分支上返回
  相同的特性；perf-tb16-strip 的结果只在它固定的两个被改动模块（`chat_text_protocol.py`、`terminus2_goal.py`）的 SHA-256 上不同。

等价性不需要 GPU 检查：八个策略在 GPU 上运行的内容都没有变化。如果要做，最小的检查是每种 kind 一次简短的 Qwen 运行（SWE-bench 冒烟
子集、16-token 页：`observation_query`、`random_fill`、`recent_tools`、`akv_rule`），各自与同种子下的 `4bf2c40` 对比：每轮 receipt 的
排序和选中页相同。

## 6. token 级选择（方案 A：标记页，方案 B：token 聚集）

组里有些方法在 8,192 token 的预算内按 token 粒度选择历史：依次保留系统提示和第一条用户消息、最近两轮（最多 1,024 个 token），然后是
Goal 分数最高的 token；在此之上，再保护协议标记（`</tool_response>`、`</think>`、`<|im_end|>`、`<tool_response>`：历史中的每一次出现，
每个请求约 106 个 token；在这些实验中它的重复率最低）和分隔符 token（空白以及 `. , ? ! ; :`，最新的，最多 1,024 个）。本节概述运行时如何
服务这类方法。它吸收了原来在 `main` 上的 token 级设计说明 `docs/token-selection/DESIGN.md`（2026-11-16 并入本节并删除）；方案 A 见第 9 节，
方案 B 见 [docs/token-gather/DESIGN.zh-CN.md](token-gather/DESIGN.zh-CN.md)。哪些运行用过它们，见 [docs/status.zh-CN.md](status.zh-CN.md)。

**本运行时的约束。**

- **页。** 运行时恢复的是 P = 16 个 token 的整页（旧布局为 784）：注意力读取被恢复页的 KV 块，receipt、记录和计划的预算都以页为单位。
  vLLM 0.24 的 FlashAttention 后端只接受 16、32、64 的块大小，所以页不能小于 16 个 token；8 或 1 个 token 一页需要换一个注意力后端。
- **层。** 选择只作用于 Qwen3.6-27B 的 16 个全注意力层。48 个 Gated-DeltaNet 层保存覆盖整段历史的循环状态，不受选择影响。
- **打分。** 每页一个向量：Goal 向量或 observation 向量与每页的向量比较。没有逐 token 的分数。
- **协议。** SWE-bench 的协议 `mswea_xml_v1` 中没有 `<tool_call>` 和 `<tool_response>` 这两种 token；标记对应到 `</think>`、
  `<|im_end|>`，以及每个 observation 开头的 `<|im_start|>user`。标记集合是参数。

**服务 token 级策略的三种做法**（policy-registry-1112 的设计说明）：

1. **把 token 映射到页（不改运行时）。** 策略给 token 打分，填充策略按聚合值（页内 token 的最大值或总和；受保护的 token 让它所在的页在
   本轮成为必选）给候选页排序。代价：选中一个 token 会带上整页，因此在 P = 16 时 token 预算 B 中只有 B/16 到 B 个是策略想要的
   token（其余是邻居）；保护 k 个分散的 token 最多花 k 页（16k 个 token）。策略需要 token 流（transcript 的 `stream`，与 recent_tools 的
   `call_start` 一样）和位置（`turn_ends`、`observation_spans`）；自 marker-pages-1114 起 FillContext 也提供 token 流
   （`FillContext.tokens`，第 9 节）。
2. **更细的页或聚集。** 8-token 页会让 block table、页向量、选页的逐页工作、receipt 的排序和恢复调用都翻倍，只换来一半的粒度，而且小于
   kernel block 时注意力后端本身就要改。精确到 token 的选择则在恢复时把选中的 KV 行聚集到连续的块中（每轮一次拷贝，保留原位置）：一条新的
   恢复路径。
3. **始终保留的区间列表。** contract 固定一组 token 模式；服务把覆盖其中任一区间的每一页都加入必选页，和初始 prompt 一样不计入预算。
   代价与做法 1 相同（每个区间一页）。

**方案 A，整页保留含标记的页**（做法 1，计入预算；marker-pages-1114，第 9 节）。除始终保留的页之外，每个含标记的候选页都通过注册表的
query 钩子受到保护，由新到旧，至多占填充空间的 `--marker-cap-percent`；预算的其余部分按基础策略以 Goal 或 observation 的顺序填充。为此
钩子新增了页的 token id、Goal 向量以及对插件固定值的启动检查。代价：在两次已记录的运行上离线统计，预算为 8,192 token：用
`observation_query`（thinking 开，作业 3539260）时标记页平均占预算的 19 %，用 `goal_query` 加 vLLM Goal rewriter（thinking 关，作业
3500565）时占 23 %，p90 为 38–44 %；每个标记会连带它所在页的 15 个邻居。分隔符没法这样保留：最新的 1,024 个分隔符分布在约 313–333
页上，占预算的 61–65 %。

**方案 B，聚集选中的 token**（P = 16 下的做法 2；token-gather-1114，`goal_query` 和 `observation_query` 上的 `--token-gather markers`
或 `separators`）。恢复时，选中 token 的 K/V 从 DRAM 中的页拷进选中页之后新的 16 token HBM 块，按逻辑顺序排列，最后一块补齐；K 在旋转
位置编码之后存储，所以聚集过去的 token 位置不变。selection contract、receipt 和证据记录 token 级的选择；恢复、phase 1 和容量统计持有聚集出来
的块，关闭回执把它们单列出来（`released.gather_blocks`）；与 `--release-idle-pages` 一起给出时被拒绝。只支持 Qwen、`mswea_xml_v1` 和
16 token 的页；Gemma 4 被拒绝。不包含：逐 token 的 Goal 打分，它需要每个 token 一个向量和一个分数（现在数量的 16 倍），以及 goal_query
工作的具体打分规则。

**A 和 B 不能组合。** `--token-gather` 只服务内置的 `goal_query` 和 `observation_query`，所以与标记插件和 `ledger_query` 一起给出时，
启动器和 selection contract 都拒绝它（第 9 节）。

**待确认的问题**（给 token 级实验）：保留哪些标记，上面的对应关系是否可以接受；保留每一次出现，还是从新到旧设上限，算在 8,192 token
预算之内还是另外加；保留每个标记所在的 16 token 页（方案 A）与只保留标记本身相比效果是否够接近（在 goal_query 的环境里做一次“保留标记所在的
16 token 块”的消融实验就能回答）；是否必须逐 token 用 Goal 打分，这些分数怎么算。

## 7. harness 的后续改动（本分支未做）

policy-registry-1112 没有改动 harness。要让计划从注册表而不是封存的源码文本中读取策略名和属性，需要：

1. `policy_list(serve)` 以及 `serve.py` / `remote_ra_http/config.py` 两行 POLICIES 的交叉检查（perf-swe32、perf-mab、perf-tb16-strip、
   perf-gemma-swe32 的 `runtime_features`）：改为读取封存源码的 `describe()`（把 `<release>/source` 放进 `sys.path` 后执行一次
   `python -I -B -c`，或用 AST 读 `source-overlay/policy_registry.py` 的 `BUILTIN` 和 `PLUGINS`）；之后这两行 POLICIES 可以删除。
2. `RUNTIME_POLICIES`、perf-mab 的 `BUDGET_POLICIES`、perf-swe32 的 `WINDOW` 和 `RULE`，以及使用它们的计划检查
   （`s['policy'] in RUNTIME_POLICIES + WINDOW + RULE + (STOCK,)`、有预算当且仅当是预算策略）：由 `describe()` 推出（`selector_v2`、
   `budget == 'fixed'`、`kind`、`budget == 'rule'`）。
3. perf-swe32 的 `window_feature` 和 `akv_feature`（`RECENT_FILL = ...`、`WINDOW_ORDERS`、`AKV_RULE_ID`、`AKV_TOOL_RESPONSES` 标记），
   以及各 controller 的 `random_fill_pin`（`DEFAULT_FILL_SEED`、`RANDOM_FILL_RNG`）：从 `describe()` 读取每个策略的 `pin` 和 `order_id`。
4. Goal 模型的检查（`'def uses_goal_...(' in contract` 和 `'goal_rewriter_use(' in serve`）：读取 `goal_engine`。
5. perf-tb16-strip：审阅后重新接受 chat-text 模块（`INTERFACE_ACCEPTED`；`chat_text_protocol.py` 的 SHA-256 因 `policies()` 和 Goal
   文本开关而改变）以及 Terminus 2 Goal 输入模块（`terminus2_goal.py`）的固定值；它的 `pin_status` 和 `chat_feature` 应按
   `chat_text_protocol.policies()`（注册表中的名字）而不是常量 `POLICIES` 检查计划；`fake_stage.py` 和 `gate_checks.py` 也同样构造
   POLICIES 列表。
6. `postrun/selection_parity.py`（perf-swe32、perf-mab、perf-gemma-swe32）：其选页策略的 `POLICIES` 元组由 `describe()`（`selector_v2`）
   得出，每个策略的记录键由 `evidence_key` 得出。
7. perf-swe32 `build.py` 的 `WINDOW_POLICIES=` 一行以及计划中的服务键：由 `describe()` 得出。
8. 所有 harness：和任何运行时改动一样，运行时固定值（commit、release receipt、模块哈希）迁移到新的运行时 commit。

## 8. query 钩子与 ledger_query（ledger-policy-1112）

ledger-policy-1112（基于 policy-registry-1112 `755550e`；用已记录的 SWE-bench 请求做了离线重放验证；它的 GPU 运行见 [docs/status.zh-CN.md](status.zh-CN.md)）。
它加入了按 query 排序的插件所需的最小钩子，以及一个发布插件。

**钩子。** query 插件是 kind `query`、使用 observation 向量、`hooked` 为真、有自己的 contract 固定值、记录写在 `hook` 下的策略。
每个生效的轮次中，Selector V2 像 `observation_query` 一样计算划分（必选页、最近页、容量）和其余候选页的余弦顺序；然后在两条选择路径上，
策略的 `query_hook(context)` 返回一个 `QueryHook`：`protected` 是先取的候选页（计入预算），`excluded` 是永不填充的候选页，
`units` 是余弦填充要么整体取、要么完全不取的页集合。`hooked_fill(order, room, hook)` 就是这个填充（Selector V2 的贪心循环，每页计 1）。
context 是填充 context 加上 `messages`（初始 prompt 之后已封存历史的 `HistoryMessage` 行：角色、token 区间、调用起点、动作文本、
reasoning，由 authority 从 XML transcript 构造：重建会话中重新编码的消息在前（transcript 现在记录它们的区间 `rebuilt`），之后是重建以来的
每个回复及其后的 observation），以及每个 session 一份用于缓存的 `state` 字典。receipt 和 `runtime.jsonl` 带有 observation_query 的字段
（包括分数）以及策略的记录 `query.selection.hook`；selector 配置哈希加入策略的顺序标识。`protocols` 把策略限制在部分启动协议上（`serves`）；
八个内置策略服务所有协议；chat-text 服务只列出服务 chat-text 的插件（`chat_text_protocol.policies()`，ledger-int-1112）。

**ledger_query**（`source-overlay/page_policies/ledger_query.py`，在 `PLUGINS` 中注册）是 observation_query 加上 2026-11-11 选页研究中的
三条规则，只服务 Qwen XML 协议（`mswea_xml_v1`；拒绝 kv-session、chat-text 和 Gemma 4 XML 语法）：

1. 文件与编辑账本：填充空间（B = 容量 - 必选页 - 最近页）中至多 `ledger_cap_percent`（默认 50，`--ledger-cap-percent`）预留给 agent
   最近一次编辑的调用，然后是每个文件自上次编辑以来最近一次读取的 observation，由新到旧。每一项要么取其全部候选页，要么一页不取；放不下的项被跳过，
   成为余弦填充中的一个单元（余弦顺序到达它且放得下时整体取，否则不取）。B 的其余部分是余弦填充。某文件最近一次读取被 harness 截断时，
   该文件自上次编辑以来最近一次完整读取也是一项。
2. 去重：填充永不选取完全落在以下区间内的页：其所有文件读取都被之后对同一文件的读取覆盖的读取的 observation（中间没有编辑；整文件查看覆盖一切，
   行区间覆盖其内的查看，被截断的输出（包括实时尾部的）不覆盖任何内容）、agent 之后原样再次运行的命令的 observation（不是纯读取的命令：
   只在两次运行之间没有编辑时）、被原样重复的回复。
3. 轻 reasoning：填充永不选取完全落在比最近两个回复更早的回复的 reasoning 内的页。

命令是回复中唯一的 `<mswea_bash_command>` 块，每个回复只解析一次：读取（cat、nl、head、tail、sed -n、awk 行区间、对文件的 grep，以及
head / tail / sed -n / grep 过滤）、编辑（写入非输出文件、sed -i、perl -i、cp / mv / rm / touch、patch、git apply / checkout 等、编辑器、
会写文件的 Python 代码及其写出的脚本）、diff（git diff / show、diff）；其他命令既不是读取也不是编辑（pip install 等环境配置不算编辑）。固定值
`{order, ledger_cap_percent, reasoning_turns: 2, dedupe: true}` 是 contract 的 `ledger_query` 键。记录包括填充空间、上限、每个账本项
（类型、token 区间、页数、预留 / 跳过 / 被整体填充 / 已被关注）、被排除的页及原因、填充计数、解析计数，以及被关注历史（初始 prompt 之外）中代码
（文件读取、diff、编辑调用）、其他工具输出、工具调用和 reasoning 的 token 占比，分别给出本次选择和同一轮纯余弦选择的占比。

**ledger-fixes-1116**（order 为 `ledger_dedupe_reasoning_light_v2`，原为 `_v1`；以 `a2f9e36` 合入 integrate-1116）。在作业 3611601
（跑的是 `_v1`）上离线重放：它的 1,208 个选择性请求中有 594 个（32 个 session 中的 22 个）选择不同（未审计）；v2 还没有在 GPU 上运行（
[docs/status.zh-CN.md](status.zh-CN.md)）；解析器仍然看不到的内容见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 4.1。来自对该策略一份评审的核查：
- 跨编辑的去重：不是纯读取的重复命令（测试、脚本运行、其他输出）只在两次运行之间没有编辑时才算重复（修复前失败测试的输出不是修复后输出的副本）；
  重复的纯读取照旧算重复（读取覆盖规则处理其文件的编辑）。
- Python 写入：单独的 `.write()` / `.writelines()`（StringIO、`sys.stdout`、astropy 表）不是文件写入，不再使命令成为未知文件的编辑
  （那会清空所有文件的读取历史）；`open(..., 'r+')` 和 `Path(...).open('w')` 是写入；代码中带引号的路径按工作目录解析，带引号的路径全是
  输出文件（.txt .log .out .diff .patch）的代码不算编辑。
- 被截断的读取：实时尾部（最新回复的输出，不在 `messages` 中）作为 `FillContext.live_observation` 传给钩子（由 authority 在 XML transcript
  的延续轮设置，否则为 None，其他策略都不读它），因此被截断的整文件重读不再取代之前的完整读取；某文件最近一次读取被截断时，自上次编辑以来
  最近一次完整读取也是账本项。
- 文件：解析后的路径只与自身匹配（仓库文件与其 site-packages 副本、两个包的 `__init__.py` 是不同文件）；`cd $VAR`（目录未知）之后读取的
  路径按后缀匹配。
- 解析器：注释；`$(( << ))` 不是 heredoc；`git -C <dir>`、`git rm` / `git mv`；pushd / popd 与 `( ... )` 子 shell 保持工作目录；
  `python -W` / `-X` 的值；`sed -i` 作用于无后缀文件；awk 严格比较（NR > a 即从 a + 1 起）与字段输出（过滤读取）；`head -c` / `tail -c`
  （过滤读取）；`sed -En` / `-nr`；`nl -n` 的值；`grep -ePAT`。
- reasoning：thinking 开启且没有 `</think>` 的回复（思考中被截断：整个回复都是 reasoning；以前 header 之后全算调用），
  `selection_contract.call_start` 返回回复末尾。这也改变了 thinking 开启时 `recent_tools` 的结果（这种回复的页是 REASONING 而不是 CALL）。

launcher（同一分支）：本包（同一 `release_id`）的 release 若安装的源文件与 checkout 的 `INSTALL_RELEASE.json` 不同，`serve.py` 拒绝启动
（launcher 导入的是 checkout 的注册表）；请从该 checkout 重新安装，或从 release 的提交运行 `serve.py`。harness 从它运行的 checkout 安装。

未完成：goal_query 变体（用 Goal 向量的钩子；marker-pages-1114 加入了 Goal 向量的钩子，见第 9 节，但 `ledger_query` 本身仍用
observation 向量）、Gemma 4 语法，以及 harness 的读取端（第 7 节：harness 读取的 `POLICIES=(...)` 行中没有 `ledger_query`）。

## 9. 标记页：goal_query_markers 与 observation_query_markers（marker-pages-1114）

marker-pages-1114（基于 ledger-int-1112 `110632b`；用已记录的 SWE-bench 请求做了离线重放验证；它的 GPU 运行见 [docs/status.zh-CN.md](status.zh-CN.md)）。
有些 token 级选择方法在固定 token 预算下、在自己的 token 排序之上，保留少数几个协议标记（`</tool_response>`、`</think>`、`<|im_end|>`、
`<tool_response>`）在历史中的每一次出现。运行时按整页（16 token，vLLM 0.24 中 FlashAttention 最小的 kernel block）恢复，所以这两个
策略保留含标记的页：第 6 节做法 1 的页近似，计入预算。

**钩子新增的内容（对所有 query 插件；钩子的约定不变）。**

- `FillContext.tokens`：会话的 token id，即引擎 transcript 自己的 `stream`（不可变 tuple，共享、不复制）。位置 [0, n·P) 是候选页目录；
  tuple 止于本轮的稳定结尾（最新 observation 的起点）；两次重建之间只增不减。`FillContext.encode`：服务 tokenizer 的文本到 id 的函数
  （不加特殊 token，解析特殊 token 文本），即 transcript 编码所用的那个函数。
- query 插件可以用 Goal 向量排序：`query_vector = 'goal'`，同时 `goal_text` 和 `goal_engine` 为真，与 `goal_query` 完全相同。注册表按
  这些属性切换的一切都像对 `goal_query` 一样作用于它：`plan_turn` 中的 Goal 前向与 Goal 缓存、Goal rewriter（必须给出
  `--goal-rewriter-model` 或 `--no-goal-rewriter`；`--goal-rewriter-batch`、`--goal-rewriter-engine vllm`、`--goal-overlap`）、goals.jsonl
  的文本以及容量门的 Goal 临时空间。`register` 拒绝没有 Goal 引擎和文本、或声明了 `rewriter_unused` 文本的 Goal 插件。
- `PagePolicy.startup(pin, encode)`：query authority 在每个服务启动时、任何请求之前调用一次，传入 contract 的固定值和 XML transcript 的
  `encode`（没有 transcript 时为 None）；插件在这里通过抛出异常拒绝自己的固定值。默认什么也不做。
- `PLUGIN_PIN_OPTIONS` 加入 `marker_set` 和 `marker_cap_percent`（`serve.py --marker-set`、`--marker-cap-percent`）。

**两个策略**（`source-overlay/page_policies/marker_pages.py`，`PLUGINS` 中两项）是 `goal_query_markers`（Goal 向量的余弦；Goal rewriter
与 `goal_query` 相同）和 `observation_query_markers`（observation 向量的余弦）。每个生效的轮次中，在 Selector V2 的划分（初始 prompt
的必选页、最近安全页、填充空间 B = 容量 - 必选页 - 最近页）之后：

1. 在已封存历史 [0, n·P) 中查找标记。标记是一个 token 序列；它的第一个 token 是标记 token，保护的是它所在的页，后面的 token 只是限定
   这次出现（`<|im_start|>user`：开启一条 user 消息的 `<|im_start|>`，即 Mini-SWE-Agent XML 协议中的一条 observation，相当于工具
   响应的开头）。落在必选页和最近页里的标记本来就会被关注，只计数。
2. 每个含标记的候选页都受保护，由新到旧，至多占 B 的 `marker_cap_percent`（不设上限：全部 B，绝不超过）。上限去掉的是最旧的标记页，
   记录中给出数量。
3. B 的其余部分是基础策略的余弦填充（`policy_registry.hooked_fill`）；被上限去掉的标记页仍可能因分数被选中。

| `--marker-set` | 标记 |
| --- | --- |
| `xml_markers`（默认） | `</think>`、`<\|im_end\|>`、`<\|im_start\|>user`（reasoning 结束、消息结束、observation 开始） |
| `ends` | `</think>`、`<\|im_end\|>` |
| `message_ends` | `<\|im_end\|>` |
| `reasoning_ends` | `</think>` |
| `tool_markers` | `</tool_response>`、`</think>`、`<\|im_end\|>`、`<tool_response>`（原生 tool-call 协议的四个标记；在 `mswea_xml_v1` 中两个 tool-response 标签从不出现） |
| 列表，例如 `"</think>,<\|im_end\|>"` | 至多 8 个互不相同的标记，每个是尖括号中的一个 token，后面可跟一个单词 |

服务启动时用自己的 tokenizer 检查每个标记：1 到 8 个 token，跟在其他文本之后、换行之前编码不变，各序列互不相同（用固定的 Qwen3.6
tokenizer，所有命名集合都通过：`</think>` 248069、`<|im_end|>` 248046、`<|im_start|>user` 248045 846、`</tool_response>` 248067、
`<tool_response>` 248066）。扫描是增量的：每轮只扫描上一轮之后新增的 token（会话的钩子 state，重建或关闭时丢弃）。

固定值 `{order: marker_pages_newest_first_v1, marker_set, markers, marker_cap_percent}` 是策略的 contract 键（`marker_cap_percent` 为
null：不设上限）。receipt 和 `runtime.jsonl` 带有基础策略的字段（分数、Goal 块）以及记录 `query.selection.hook`：固定值、`room_pages`、
`cap_pages`、`occurrences`（每个标记的出现次数）、`marker_tokens`（已封存历史中不同的标记位置数）、`fixed_tokens`（其中落在必选页和
最近页的）、`marker_pages`（含标记的候选页）、`protected_pages`、`dropped_pages`、`dropped_tokens`、`dropped_filled`（被去掉、但仍被余弦
填充选中的页）、`kept_tokens`（落在选中页上的标记）、`budget_share`（受保护页 / 容量）、`room_share`、`new_pages`（纯余弦填充本不会
选的受保护页）、`displaced_pages`（被挤出的纯余弦页）、`fill`（hooked_fill 的计数）、`scanned_tokens` 和 `plain_jaccard`。

拒绝：kv-session 和 chat-text 协议以及 Gemma 4 profile（`protocols` 只有 `mswea_xml_v1`；启动器和启动配置检查）、1-100 以外的上限、
未知的标记集合或格式错误的列表（启动器）、tokenizer 无法按上述方式编码的标记（服务启动时）、与其他任何策略一起给出的
`--marker-set` / `--marker-cap-percent`，以及与任一标记策略一起给出的 `--token-gather`（做法 B，token-gather-1114，只服务内置的
`goal_query` 和 `observation_query`；启动器和 contract 都拒绝，integrate-1114）。基于非 query 策略（kind 为 `window`、`rule`
或 `random` 的填充策略）的标记插件无法注册：query 钩子只属于 query 插件。

**离线重放（未审计）。** 在两次 SWE-bench 运行的已记录请求上运行已实现的代码，预算 8,192 token（512 个 16-token 页），每个请求用它
自己的划分、余弦顺序和 token 流，以及固定的 Qwen3.6 tokenizer：job 3539260（`observation_query`，thinking 开；1,198 个选择性请求）
用 `observation_query_markers`，job 3500565（`goal_query` 加 vLLM Goal rewriter，thinking 关；1,677 个）用 `goal_query_markers`。
不加钩子时重放复现了每一个已记录的选择；增量扫描在每个请求上都与完整扫描一致。默认集合 `xml_markers`；页的占比以 512 页预算计：

| | 3539260（observation） | 3500565（Goal） |
| --- | --- | --- |
| 必选页和最近页之外的标记 token，均值 | 129 | 180 |
| 含标记的候选页：均值（中位数、p90、最大值） | 97（80、196、345） | 117（103、223、304） |
| 它们占预算：均值 / p90 / 最大值 | 18.9 / 38.3 / 67.4 % | 22.9 / 43.6 / 59.4 % |
| 标记页超过填充空间的请求（不设上限：去掉最旧的） | 11 | 33 |
| 不设上限：受保护页上的标记；被挤出的余弦页，均值（占预算） | 99.8 %；51（9.9 %） | 99.4 %；98（19.1 %） |
| 上限 50：受保护页占预算均值 / 最大值；受保护页上的标记 | 16.5 / 34.4 %；86.8 % | 19.7 / 39.8 %；85.9 % |
| 上限 25：受保护页占预算均值 / 最大值；受保护页上的标记 | 12.1 / 19.7 %；63.2 % | 13.4 / 20.1 %；58.5 % |
| 选中的标记：基础策略 / 不设上限 / 上限 50 / 上限 25 | 52.7 / 99.9 / 90.3 / 76.5 % | 20.4 / 99.5 / 86.9 / 62.8 % |
| 与基础选择的页集合 Jaccard，均值（不设上限） | 0.77 | 0.61 |
| 每个请求的钩子 + 填充 CPU 时间：中位数 / p95 / 最大值 | 0.22 / 0.44 / 4.75 ms | 0.24 / 0.45 / 1.20 ms |

每个标记会连带同页的 15 个邻居：每页 1.3-1.5 个标记，页的代价约为 token 代价的 12 倍（129-180 个 token 只占预算的 1.6-2.2 %）。重放
固定了 agent 的行为；对 agent 的影响只有 GPU 运行才能看出。job 3576565（`goal_query`，thinking 开）未重放：本地记录中没有它的 prompt。

**等价性。** `tools/policy_parity.py` 现在覆盖本分支之前的九个策略（八个内置策略，以及带固定值和 `--ledger-cap-percent` 的
`ledger_query`）：与 `110632b` 相比，12,259 个结果中 12,257 个逐字节相同；另外两个（`serve.py --help` 和未知策略的用法错误）只差两个
新的 `--policy` 选项和两个新参数。

**未完成。** 分隔符（最新 1,024 个分隔符约占 310-330 页，在 36-54 % 的请求中超过填充空间）；逐 token 的 Goal 分数（每页只有
一个向量）；Gemma 4 profile（范围：先做 Qwen3.6-27B）；harness 的读取端（第 7 节：harness 读取的 `POLICIES=(...)` 行中没有这两个新策略）。
