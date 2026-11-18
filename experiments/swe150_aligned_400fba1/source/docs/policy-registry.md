# Page-selection policy registry

English | [简体中文](policy-registry.zh-CN.md)

policy-registry-1112 (on integrate-1111 `4bf2c40`; the GPU runs of its policies, where they have run, are in [docs/status.md](status.md)). Every page-selection policy is declared once, in
`source-overlay/policy_registry.py`, and every other part of the runtime reads policy names and properties from
there. A new page-level policy is one class plus one registration (the README's section "How to add a page-selection
policy"). The CLI and HTTP surface, the startup records, the contracts and their digests, the refusals, the
receipts and the evidence keys and values of the eight policies are those of `4bf2c40` (section 5).

1. [Inventory](#1-inventory): where the runtime named or special-cased a policy at `4bf2c40`
2. [The registry](#2-the-registry): the properties, the eight policies, the API, the fill path for plug-ins
3. [What each module reads now](#3-what-each-module-reads-now)
4. [Kept for compatibility](#4-kept-for-compatibility)
5. [Parity evidence](#5-parity-evidence)
6. [Token-level selection (option A: marker pages, option B: token gather)](#6-token-level-selection-option-a-marker-pages-option-b-token-gather)
7. [Harness follow-ups (not done here)](#7-harness-follow-ups-not-done-here)
8. [The query hook and ledger_query (ledger-policy-1112)](#8-the-query-hook-and-ledger_query-ledger-policy-1112)
9. [Marker pages: goal_query_markers and observation_query_markers (marker-pages-1114)](#9-marker-pages-goal_query_markers-and-observation_query_markers-marker-pages-1114)

## 1. Inventory

Every place of the runtime (not the tests, not `harnesses/`) that named a policy at `4bf2c40`, found by an AST scan
of every `.py` file of `serve.py`, `tools/` and `source-overlay/` (string constants equal to a policy name, texts
naming one, references to the policy constants) and read site by site. Categories: **S** selection logic,
**B** budget / capacity, **F** feature flags (Goal text, Goal engine, the window / AKV / fill records, the thinking
interplay), **V** argument validation, **R** records / metrics, **D** docs (help texts, messages).

| Category | Sites | Files |
| --- | ---: | --- |
| V argument validation | 27 | serve.py, remote_ra_http/config.py, selection_contract, query_selection, engine_boundary, private_endpoint, controller, chat_text_protocol, the native adapter, the private native endpoint, frontend_registry, registered_add_bridge, live_frontend, native_authority |
| F feature flags | 26 | serve.py, remote_ra_http/config.py, evidence, chat_text_route, selection_contract, query_selection, private_endpoint, phase1_coordinator, window_phase1, goal_overlap, the Goal batch and vLLM-engine checks, terminus2_goal, chat_text_protocol |
| S selection logic | 16 | query_selection, selection_contract, phase1_coordinator, window_phase1, the native select_pages, remote_ra_growing |
| B budget / capacity | 12 | serve.py, remote_ra_http/config.py, selection_contract, query_selection, capacity, window_capacity |
| R records / metrics | 7 | serve.py, evidence, selection_contract, query_selection, kv_session_protocol |
| D docs | 2 + 16 files | serve.py's help texts, tools/step_profile.py's labels; 16 modules that name policies in comments, docstrings or log texts only |
| **Live path** | **90 sites** | **30 files** (28 modules of the live path, the vLLM overlay's copy of `selective_policy.py`, `tools/step_profile.py`) |

Out of scope, unchanged: the legacy development layers that only know `full`, `random50`, `goal_query` and
`observation_query` and are not on the live selection path (`policy_runtime/`, `swebench_baseline/`,
`dynamic_runtime.py`, `goal_adapter.py`, the PoC policy adapter `*_vllm_policy_adapter.py`,
`remote_ra_growing/controller.py`, `remote_ra_linux/live_selector.py` and `live_factory.py`,
`vllm/v1/worker/goal_producer_bridge.py`, the offloading scheduler's legacy ranking plan): 21 files.

The sites (file, site, category: what it named at `4bf2c40` → what it reads now):

| # | File | Site | Cat. | At `4bf2c40` → now |
| ---: | --- | --- | :---: | --- |
| 1 | serve.py | `POLICIES`, `QUERY_POLICIES`, `WINDOW_POLICIES`, `AKV_RULE` | V | literal lists → kept (section 4), equal to the registry's built-in view |
| 2 | serve.py | `main`: `--policy` choices | V | `POLICIES` → `names()` |
| 3-4 | serve.py | `prepare`, `prepare_gemma4`: known policy | V | `in POLICIES` → `known()` |
| 5 | serve.py | `selection_contract`: Selector V2 gate | V | `QUERY_POLICIES` → `selector_v2` |
| 6 | serve.py | `selection_contract`: `--fill-seed` | V | `== 'random_fill'` → `pin_options` |
| 7 | serve.py | `selection_contract`: the random-fill, window and AKV pins | B | three branches → `contract_key`, `pin()` |
| 8 | serve.py | `selection_contract`: no budget options, no budget for `akv_rule` | B | `== AKV_RULE` → `budget == 'rule'`, `budget_refusal()` |
| 9 | serve.py | `goal_rewriter_use`: why a given Goal model is unused | R | four texts by name → `rewriter_unused` |
| 10-13 | serve.py | `--goal-rewriter-batch`, `--goal-rewriter-engine vllm`, `--goal-overlap` (Qwen and Gemma) | F | `== 'goal_query'` → `goal_engine` |
| 14 | serve.py | `--thinking-history strip` refuses `recent_tools` and `akv_rule` | F | name list → `reply_roles` (the text joins their names) |
| 15-16 | serve.py | `chat_text_goal` record (Qwen and Gemma) | F | `== 'goal_query'` → `goal_text` |
| 17 | serve.py | `prepare_gemma4`: the Goal transient G | B | `== 'goal_query'` → `query_vector == 'goal'` |
| 18 | serve.py | `chat_text_tasks` | V | `chat.POLICIES` → `chat.policies()` |
| 19 | serve.py | help texts, the Selector V2 option group's title | D | unchanged |
| 20 | remote_ra_http/config.py | `POLICIES`, `QUERY_POLICIES` | V | kept (section 4) |
| 21-22 | remote_ra_http/config.py | `load`: known policy; Selector V2 contract gate | V | lists → `known()`, `selector_v2` |
| 23 | remote_ra_http/config.py | `admission`: `goal_overlap` | F | `goal_engine` |
| 24 | remote_ra_http/config.py | `window_layout`: the Goal transient | B | `query_vector == 'goal'` |
| 25 | remote_ra_http/evidence.py | `write_ready`: `selection_fraction` 0.5 | R | `== 'random50'` → `budget == 'half'` |
| 26 | remote_ra_http/evidence.py | `complete`: the query row | R | six names → `selector_v2` |
| 27 | remote_ra_http/evidence.py | `complete`: `query.selection.fill` / `.window` | F | names → `evidence_key` |
| 28 | remote_ra_http/chat_text_route.py | `frontend`: the Goal inputs | F | `goal_text` |
| 29 | remote_ra_live_v2/selection_contract.py | `RANDOM_FILL` ... `AKV_TOOL_RESPONSES`, `FILL_POLICIES`, `OPTIONAL_KEYS` | V | kept (section 4) |
| 30 | selection_contract.py | `default_contract`: one keyword per pin; no budget, threshold 0 for the rule | B | four keywords → `**pins` over `pinned()`, `budget == 'rule'` |
| 31 | selection_contract.py | `validate`: optional keys, each pin, one pin, the rule's threshold | V | names → `pinned()`, `validate_pin` |
| 32 | selection_contract.py | the Goal-engine check (`uses_goal_...`) | F | `== 'goal_query'` → `goal_engine` |
| 33 | selection_contract.py | `for_policy`: a pin required or foreign | V | three checks → `pin_required()`, `pin_foreign()` per pinned policy |
| 34 | selection_contract.py | `capacity_pages`: no budget for the rule | B | `budget == 'rule'` |
| 35 | selection_contract.py | `random_fill_pin`, `window_pin`, `akv_pin` and their validators | S | kept: the built-in pins, reached from the registry's classes |
| 36 | selection_contract.py | `window_order` | S | kept: the built-in orders, called from `select()` |
| 37 | selection_contract.py | `window_record` | R | kept (+ `order_id` for a plug-in), called from `record()` |
| 38 | selection_contract.py | `akv_kept`, `window_classes` | S | kept, called from `page_classes()` |
| 39 | remote_ra_live_v2/query_selection.py | `__init__`: the server's policy | V | eight names → `known()` |
| 40 | query_selection.py | `__init__`: the default contract of a pinned policy | B | three branches → `contract_key`, `pin()` |
| 41-42 | query_selection.py | `__init__`: the Goal batch / engine checks; the Goal-text self-check | F | `goal_engine`; `goal_text` |
| 43 | query_selection.py | `plan`: the rule's capacity; random_fill; the windows; the query policies | S | names → `budget`, `kind` |
| 44 | query_selection.py | `plan_reference`, `plan_fast`: `query_source` | S | `query_vector` |
| 45-46 | query_selection.py | `window_fill_selector`: its `fill` config value; its order | R, S | names → `order_id`; `fill(recent, candidates)` (the policy's `select`) |
| 47 | query_selection.py | `window_inputs`: the AKV kept Pages; recent_tools' classes | S | names → `page_classes()` |
| 48-50 | query_selection.py | `plan_window_fast` / `_reference`: the order; the rule's invariant; the record | S, B, R | names → `select()`, `budget == 'rule'`, `record()` |
| 51-52 | query_selection.py | `_begin`: the Selector V2 gate; the chat-text Goal inputs | V, F | `selector_v2`; `goal_text` |
| 53 | query_selection.py | `_finish`: the Goal cache and forward | F | `query_vector == 'goal'` |
| 54-56 | query_selection.py | `plan_turn`: no query vector; the Goal vector; the Goal texts | S, S, F | `FILL_POLICIES`, names → `query_vector`, `goal_text` |
| 57 | query_selection.py | `random_fill_selector`, `plan_random_fill_*` | S | kept: random_fill's own (kind `random`) |
| 58-60 | remote_ra_live_v2/capacity.py | `QUERY`; `selected_max`; the Goal Pages of `_need_by_pool` | B | names → `budget`, `selector_v2`; `query_vector` |
| 61 | remote_ra_live_v2/engine_boundary.py | `QUERY`: the receipt check | V | `selector_v2` |
| 62-63 | remote_ra_live_v2/private_endpoint.py | `QUERY`: the authority; `--goal-overlap` | V, F | `selector_v2`; `goal_engine` |
| 64 | remote_ra_live_v2/controller.py | `POLICIES` | V | `known()` |
| 65-67 | remote_ra_live_v2/phase1_coordinator.py | a Goal forward (twice); the observation's query tokens | F, S, F | `query_vector` |
| 68-70 | remote_ra_live_v2/window_phase1.py | the same three | F, S, F | `query_vector` |
| 71 | remote_ra_live_v2/window_capacity.py | `_gate_turn`: the Goal transient | B | `query_vector == 'goal'` |
| 72-74 | goal_overlap.py, goal_rewriter_batch.py, goal_rewriter_vllm.py | the Goal-engine checks | F | `goal_engine` |
| 75 | remote_ra_live_v2/terminus2_goal.py | `check_config` | F | `goal_text` |
| 76-77 | remote_ra_live_v2/chat_text_protocol.py | `POLICIES` and `check_config`; `options`' Goal inputs | V, F | `policies()` (section 4); `goal_text` |
| 78 | remote_ra_live_v2/kv_session_protocol.py | `options`: `fill_seed`, `fill_order`, `budget_mode` `rule` | R | names → each pin's `info()` |
| 79-80 | remote_ra_native_growing/adapter.py | `QUERY` and `open`; `reserve` / `complete` | V | `known()`; `selector_v2` |
| 81 | remote_ra_native_growing/contract.py | `select_pages`: full, random50, Selector V2 | S | names → `kind`, `budget == 'half'` (`select_native`) |
| 82 | remote_ra_private_v2/lifecycle.py | the native endpoint admits full and random50 only | V | `not selector_v2` |
| 83-87 | frontend_registry.py (2), registered_add_bridge.py, remote_ra_linux/live_frontend.py, native_authority.py | the routing tag, the session, the startup | V | eight names → `known()` |
| 88 | remote_ra_growing/contract.py | `select_random50` | S | kept: random50's draw (`RandomPageSelectionPolicy` at fraction 1/2), reached from the registry |
| 89 | remote_ra_growing/selective_policy.py (and its vLLM overlay copy) | `POLICY_NAMES` of the request-contract v2 path | S | kept (the PageSelectionPolicy protocol the registry builds on) |
| 90 | tools/step_profile.py | profile labels naming `goal_query` | D | unchanged |

The docs-only modules: `admission_flags`, `admission_overlap`, `composition`, `deferred_completion`, `goal_async`
(also one abort text), `goal_text`, `kv_layout`, `lifecycle`, `model_profile`, `quoted_literals`, `registered_route`, `gemma4_renderer`,
`engine_assembly`, `registered_add_protocol` and the vLLM engine core (`core.py`, `core_client.py`).

## 2. The registry

`source-overlay/policy_registry.py` is a top-level module that imports nothing outside the standard library at
import time: `serve.py` reads it before the release is on `sys.path` (from the release source when importable, else
from its own `source-overlay`). Each policy is one `PagePolicy` subclass; `register(Class())` adds it.

### 2.1 Properties

| Property | Values | Read by |
| --- | --- | --- |
| `name` | lower_snake_case | `--policy`, the startup config, every gate |
| `kind` | `full`, `random`, `query`, `window`, `rule` | the plan's dispatch, `select_pages` |
| `selector_v2` | bool: a selection contract, two-phase query turns, a signed query receipt | the launcher, config, engine, endpoint, adapter, evidence |
| `query_vector` | `None`, `'observation'`, `'goal'` | phase 1's query tokens, the Goal forward and cache, `query_source`, the Goal transient |
| `goal_text` | bool: the query is the Goal's text | goals.jsonl texts, the Terminus 2 Goal inputs (`chat_text_goal`), the Goal-text self-check |
| `goal_engine` | bool: runs the Goal rewriter | `--goal-rewriter-model` pinned or recorded unused, `--goal-rewriter-batch`, `--goal-rewriter-engine vllm`, `--goal-overlap` |
| `budget` | `none`, `half`, `fixed`, `rule` | `selected_max`, `capacity_pages`, `default_contract`, the launcher's budget refusal, `selection_fraction` |
| `contract_key`, `pin()`, `validate_pin()`, `pin_options` | the policy's own key in the selection contract (its name) or None | `default_contract`, `validate`, `for_policy`, the launcher, the kv-session info (`info()`) |
| `evidence_key` | `fill`, `window`, `hook` (a query plug-in, section 8) or None | the plan's evidence, the receipt's `selection`, `runtime.jsonl`'s `query.selection` |
| `reply_roles` | bool: tells a reply's tool call and reasoning from tool responses | `--thinking-history strip`'s refusal |
| `rewriter_unused` | text | startup.json's `goal_rewriter.reason` |
| `order_id` | text | the selector config's `fill`, the record's `order`, a fill policy's pin `{order}` |
| `protocols` | None (every protocol) or the startup protocols served (ledger-policy-1112) | the launcher's and the startup config's refusal (`serves`, `protocol_refusal`) |
| `hooked` | bool: a query plug-in, whose cosine fill runs through `query_hook` (section 8; the Goal vector since marker-pages-1114, section 9) | the plan (`plan_fast`, `plan_reference`), the query authority's startup check (`startup`) |

A fill policy (kinds `window` and `rule`) also has `select(context)`, `page_classes(view)` and `record(context,
selected, order, filled)` (section 2.3).

### 2.2 The eight built-in policies

| Policy | kind | selector_v2 | query_vector | goal_text | goal_engine | budget | contract_key | evidence_key | reply_roles | order_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | full | no | – | no | no | none | – | – | no | – |
| `random50` | random | no | – | no | no | half | – | – | no | – |
| `goal_query` | query | yes | goal | yes | yes | fixed | – | – | no | – |
| `observation_query` | query | yes | observation | no | no | fixed | – | – | no | – |
| `random_fill` | random | yes | – | no | no | fixed | `random_fill` (`fill_seed`) | `fill` | no | `splitmix64_fisher_yates_v1` |
| `recent_fill` | window | yes | – | no | no | fixed | `recent_fill` | `window` | no | `newest_first_v1` |
| `recent_tools` | window | yes | – | no | no | fixed | `recent_tools` | `window` | yes | `tool_calls_then_tool_responses_then_reasoning_newest_first_v1` |
| `akv_rule` | rule | yes | – | no | no | rule | `akv_rule` | `window` | yes | `replies_and_latest_tool_responses_v1` |

They are registered in `4bf2c40`'s order (`BUILTIN`). Their algorithms stay where they were and are reached from
their classes: the pins, `window_order`, `window_record`, `akv_kept` and `window_classes` in
`remote_ra_live_v2/selection_contract.py`, random50's draw in `remote_ra_growing/contract.py`
(`RandomPageSelectionPolicy` of `selective_policy.py`), random_fill's selector and plan in `query_selection.py`, the
cosine ranking of the query policies in the upstream Selector V2.

### 2.3 The fill path (plug-ins)

A plug-in is a fill policy of kind `window` (the fixed budget) or `rule` (no budget: its `page_classes` returns the
Pages it keeps as its first set, as `akv_rule`). Every active turn runs Selector V2's boundary, mandatory Pages (the
initial prompt) and recent safety Pages, then calls the policy:

- `select(context)` on the PageSelectionPolicy protocol of `remote_ra_growing/selective_policy.py`: it gets a
  `FillContext` and returns a `PageSelection` whose `logical_pages` are the fill order, a permutation of
  `context.candidates` (checked); the server takes the first `capacity - mandatory - len(recent)` of them. The
  context holds `num_pages`, `page_tokens`, `mandatory`, `recent`, `candidates`, `capacity`, `local_turn`, `turn`
  (the request ordinal), `session_index`, `turn_ends`, `observation_spans`, `stable_end`, the two Page sets of
  `page_classes`, the contract's `pin` and the candidate Pages' vectors (`page_vectors`);
- `page_classes(view)`: two Page sets for `select` and `record` (default: empty);
- `record(...)`: the policy's record under `window` (default: `window_record`'s common fields with its `order_id`).

The plug-in's contract carries `{order: order_id}` under its name (`for_policy` refuses it missing, or carried by
another policy's contract); the launcher accepts `--active-ratio` / `--active-token-budget` (kind `window`) and
refuses `--fill-seed`; a given `--goal-rewriter-model` is recorded unused. A plug-in may not declare a query vector,
the Goal text or engine, pin options or another evidence key (`register` refuses them).

A query plug-in (kind `query`, the observation vector's cosine with a hook on the fill) has its own path since
ledger-policy-1112 (section 8); since marker-pages-1114 it may rank by the Goal vector instead, as `goal_query`
(section 9). Not supported yet: a plug-in with a new score over the Page vectors, and a plug-in of its own kind
`random` (it needs its own selector config as random_fill's).

### 2.4 Registration and API

- `register(policy)`, `unregister(name)` (plug-ins only), `registered(policy)` (a block, for tests);
  `PLUGINS = ('module:Class', ...)` and `load(entries)`: a release plug-in is one class in its own module under
  `source-overlay/page_policies/` plus one entry in `PLUGINS`;
- `get(name)` (None for any other value), `known`, `names`, `policies`, `where(**properties)`,
  `builtin_where(**properties)`, `pinned()`, and the gates' short forms `selector_v2`, `query_vector`, `goal_text`,
  `goal_engine`, `reply_roles`;
- `describe()`: every policy as a JSON row (name, builtin, the properties, the default pin), for readers outside the
  runtime (section 7).

## 3. What each module reads now

Section 1's table, column "now". In short: name lists became `known()`; the Selector V2 sets (`QUERY`,
`QUERY_POLICIES`, `FILL_POLICIES`) `selector_v2` and `query_vector`; every `== 'goal_query'` `goal_engine`,
`goal_text` or `query_vector == 'goal'` by what it switches; the pins `pinned()` with each policy's `pin()`,
`validate_pin()`, `pin_required()`, `pin_foreign()` and `info()`; the budgets `budget`; the records `evidence_key`
and `record()`; the plan's dispatch `budget` and `kind`. Refusal texts are built from the policies' own texts, so
they are the same bytes for the eight policies.

## 4. Kept for compatibility

- `serve.py` `POLICIES=(...)`, `QUERY_POLICIES`, `WINDOW_POLICIES`, `AKV_RULE` and `remote_ra_http/config.py`
  `POLICIES=(...)`, `QUERY_POLICIES`: the harnesses read the `POLICIES=(...)` lines from the sealed source (section
  7). `chat_text_protocol.POLICIES`: the chat-text refusal's order; `policies()` appends plug-ins.
- `selection_contract`'s `RANDOM_FILL`, `RECENT_FILL`, `RECENT_TOOLS`, `WINDOW_ORDERS`, `WINDOW_POLICIES`,
  `AKV_RULE`, `AKV_RULE_ID`, `AKV_TOOL_RESPONSES`, `RANDOM_FILL_RNG`, `DEFAULT_FILL_SEED`, `FILL_POLICIES`,
  `OPTIONAL_KEYS` and its function names (harness markers, tests).
- `QUERY` of `capacity`, `engine_boundary`, `private_endpoint` and the native adapter, `controller.POLICIES`, and
  `serve.py`'s `GOAL_REWRITER_UNUSED*` texts: now the registry's view of the eight policies.

`tests/test_policy_registry.py` checks each against the registry, and an AST scan checks that no live-path module
names a policy elsewhere. These lists describe the eight policies of `4bf2c40`; plug-ins appear only in the
registry, so adding one never touches them.

## 5. Parity evidence

- **Cross-tree probe** (`tools/policy_parity.py`, run on the Git export of `4bf2c40` and of the branch with one work
  directory): 9,524 outcomes, every one byte-identical (the two JSON files have the same SHA-256): the contracts of
  P x vector format x ratio / budget x variant x Goal model x every pin, with `for_policy` of every policy on each;
  `validate` and `for_policy` on 21 malformed contracts and pin mixes; the pins; `capacity_pages`; `selected_max`
  and `select_pages` over policies x n x turns; the kv-session record; the launcher's selection contract and Goal
  record for every policy x 14 option sets x 3 layouts; `serve.prepare` for every policy x 7 option sets x 12
  protocol / flag sets (Qwen) and x 6 x 5 (Gemma 4), with `config.load`, the kv-session info and the chat-text
  options and check of every launched config; the dry-run `main` and `--help`; test_decouple_e2e's stage-B script
  for full, random50, goal_query and observation_query at P = 784 and 16 (semantics, allocations, RPC trace,
  runtime rows).
- **Tests** (unchanged pins): test_recent_fill's 144 default contracts and its recent_fill / recent_tools contracts,
  receipts and stage-B outputs, test_random_fill's and test_akv_rule's, test_random_fill_legacy's former-code
  equivalence (its AST check now undoes the registry lookups first, `REGISTRY_LATER`), test_selection_contract,
  test_selection_fast, test_decouple_selection and the release check. The literal-list checks of test_recent_fill
  and test_akv_rule now cover the lists that remain.
- **The harnesses' readers**: `runtime_features` of perf-swe32, perf-mab and perf-gemma-swe32 (their `4bf2c40`
  code) return the same features on the branch; perf-tb16-strip's differ only in the SHA-256 of the two edited
  modules it pins (`chat_text_protocol.py`, `terminus2_goal.py`).

A GPU check is not needed for parity: nothing the GPU runs changed for the eight policies. If one is wanted, the
smallest is one short Qwen run per kind on the SWE-bench smoke subset at 16-token pages (`observation_query`,
`random_fill`, `recent_tools`, `akv_rule`, each against `4bf2c40` with the same seed: the same receipts' rankings
and selected Pages per turn).

## 6. Token-level selection (option A: marker pages, option B: token gather)

Some group members' methods select history at token granularity under an 8,192-token budget: they keep the system
prompt and the first user message, then the last two turns (up to 1,024 tokens), then the tokens with the highest Goal
scores, and on top of that they protect protocol markers (`</tool_response>`, `</think>`, `<|im_end|>`,
`<tool_response>`: every earlier occurrence, about 106 tokens per request; the lowest repeat rate in those experiments)
and separator tokens (whitespace and `. , ? ! ; :`, the newest up to 1,024). This section is the overview of how the
runtime serves such methods. It takes in the token-level design note that was `docs/token-selection/DESIGN.md` on
`main` (merged here and removed on 2026-11-16); option A is section 9, option B [docs/token-gather/DESIGN.md](token-gather/DESIGN.md).
Which runs used them: [docs/status.md](status.md).

**Constraints in this runtime.**

- **Pages.** The runtime restores whole Pages of P = 16 tokens (784 at the legacy layout): attention reads the
  restored Pages' KV blocks, and the receipt, the record and the plan's budget are in Pages. vLLM 0.24's FlashAttention
  backend accepts block sizes of 16, 32 and 64 only, so a Page cannot shrink below 16 tokens; 8- or 1-token Pages would
  need another attention backend.
- **Layers.** Selection acts on the 16 full-attention layers of Qwen3.6-27B. The 48 Gated-DeltaNet layers carry a
  recurrent state over the whole history, which selection does not touch.
- **Scores.** One vector per Page: the Goal or the observation vector against each Page's vector. There are no
  per-token scores.
- **Protocol.** The SWE-bench protocol `mswea_xml_v1` has no `<tool_call>` or `<tool_response>` tokens; the markers map
  to `</think>`, `<|im_end|>` and the `<|im_start|>user` that opens each observation. The marker set is a parameter.

**Three ways to serve a token-level policy** (the design note of policy-registry-1112):

1. **Map tokens to Pages (no runtime change).** The policy scores tokens and a fill policy orders the candidate
   Pages by an aggregate (the maximum or the sum over a Page's tokens; a protected token makes its Page mandatory
   for the turn). Cost: a selected token brings its whole Page, so at P = 16 a token budget B holds between B/16 and
   B tokens of what the policy wanted (the rest are neighbours); protecting k scattered tokens costs up to k Pages
   (16k tokens). The policy needs the token stream (the transcript's `stream`, as recent_tools' `call_start`) and
   positions (`turn_ends`, `observation_spans`); since marker-pages-1114 the FillContext gives the stream too
   (`FillContext.tokens`, section 9).
2. **Finer Pages or a gather.** Pages of 8 tokens would double the block tables, the Page vectors, the selection's
   per-Page work, the receipt's ranking and the restore calls, for half the granularity, and below the kernel block the
   attention backend itself would have to change. A token-exact selection instead gathers the selected KV rows into
   contiguous blocks at restore (a copy per turn, the original positions kept): a new restore path.
3. **An always-keep span list.** The contract pins a list of token patterns; the server adds every Page that covers
   one to the mandatory Pages, outside the budget, as the initial prompt's. Cost: as way 1 (a Page per span).

**Option A, keep whole marker pages** (way 1, inside the budget; marker-pages-1114, section 9). Every candidate Page
that holds a marker, outside the always-kept Pages, is protected through the registry's query hook, newest first, up
to `--marker-cap-percent` of the room; the rest of the budget is the base policy's fill in Goal or observation order.
For it the hook gained the Pages' token ids, the Goal vector and a startup check of the plug-in's pin. Cost, offline on
two recorded runs at an 8,192-token budget: the marker Pages take 19 % of the budget on average with
`observation_query` (thinking on, job 3539260) and 23 % with `goal_query` and the vLLM Goal rewriter (thinking off,
job 3500565), p90 38–44 %; each marker brings its 15 Page neighbours. Separators cannot be kept this way: the newest
1,024 separators span about 313–333 Pages, 61–65 % of the budget.

**Option B, gather the selected tokens** (way 2 at P = 16; token-gather-1114, `--token-gather markers` or
`separators` on `goal_query` and `observation_query`). At restore the K/V of the selected tokens are copied from the
DRAM Pages into fresh 16-token HBM blocks after the selected Pages, in logical order, the last block padded; K is
stored after the rotary embedding, so the gathered tokens keep their positions. The selection contract, the receipts
and the evidence record the token picks; restore, phase 1 and the capacity accounting hold the gathered blocks, and
the close receipt itemizes them (`released.gather_blocks`); `--release-idle-pages` is refused with it. Qwen with
`mswea_xml_v1` and 16-token Pages only; Gemma 4 is refused. Not covered: per-token Goal scoring, which needs a vector
and a score for every token (16 times today's count) and the exact scoring rule of the goal_query work.

**A and B do not combine.** `--token-gather` serves the built-in `goal_query` and `observation_query` only, so the
launcher and the selection contract refuse it with the marker plug-ins and with `ledger_query` (section 9).

**Open questions** (for the token-level experiments): which markers to keep, and whether the mapping above is
acceptable; every occurrence or the newest first with a cap, inside the 8,192-token budget or on top of it; whether
keeping each marker's 16-token Page (option A) is close enough to keeping the marker alone (an ablation "keep the
marker's 16-token block" in the goal_query setup would answer it); whether per-token Goal scoring is required, and how
its scores are computed.

## 7. Harness follow-ups (not done here)

policy-registry-1112 left the harnesses unchanged. What would let plans read policy
names and properties from the registry instead of the sealed source text:

1. `policy_list(serve)` and the `serve.py` / `remote_ra_http/config.py` POLICIES cross-check
   (`runtime_features` of perf-swe32, perf-mab, perf-tb16-strip, perf-gemma-swe32): read `describe()` of the sealed
   source instead (one `python -I -B -c` with `<release>/source` on `sys.path`, or an AST read of `BUILTIN` and
   `PLUGINS` in `source-overlay/policy_registry.py`); then the two POLICIES lines can go.
2. `RUNTIME_POLICIES`, perf-mab's `BUDGET_POLICIES`, perf-swe32's `WINDOW` and `RULE`, and the plan checks that use
   them (`s['policy'] in RUNTIME_POLICIES + WINDOW + RULE + (STOCK,)`, `budget is not None` iff a budget policy):
   derive from `describe()` (`selector_v2`, `budget == 'fixed'`, `kind`, `budget == 'rule'`).
3. perf-swe32's `window_feature` and `akv_feature` (the `RECENT_FILL = ...`, `WINDOW_ORDERS`, `AKV_RULE_ID`,
   `AKV_TOOL_RESPONSES` markers) and `random_fill_pin` of every controller (`DEFAULT_FILL_SEED`, `RANDOM_FILL_RNG`):
   read each policy's `pin` and `order_id` from `describe()`.
4. The Goal-model check (`'def uses_goal_...(' in contract` and `'goal_rewriter_use(' in serve`): read `goal_engine`.
5. perf-tb16-strip: re-accept the chat-text module after review (`INTERFACE_ACCEPTED`, `chat_text_protocol.py`'s
   SHA-256 changed: `policies()` and the Goal-text switch) and the Terminus 2 Goal-input module's pin
   (`terminus2_goal.py`); its `pin_status` and `chat_feature` should check plans against
   `chat_text_protocol.policies()` (the registry's names) rather than the `POLICIES` constant; `fake_stage.py`
   and `gate_checks.py` build their POLICIES lists the same way.
6. `postrun/selection_parity.py` (perf-swe32, perf-mab, perf-gemma-swe32): its `POLICIES` tuple of selective
   policies from `describe()` (`selector_v2`), with the record key per policy from `evidence_key`.
7. perf-swe32 `build.py`'s `WINDOW_POLICIES=` line and the plans' server keys: from `describe()`.
8. Every harness: the runtime pin (commit, release receipt, module hashes) moves to the new runtime commit as for any
   runtime change.

## 8. The query hook and ledger_query (ledger-policy-1112)

ledger-policy-1112 (on policy-registry-1112 `755550e`; verified offline by a replay of recorded SWE-bench requests;
its GPU runs are in [docs/status.md](status.md)). It adds the smallest hook a query-ranked plug-in needs and one release plug-in.

**The hook.** A query plug-in is kind `query` with the observation vector, `hooked`, its own contract pin and its
record under `hook`. On every active turn Selector V2 computes the partition (mandatory and recent Pages, the
capacity) and the cosine order of the remaining candidates as for `observation_query`; then, on both selection paths,
the policy's `query_hook(context)` returns a `QueryHook`: `protected` candidate Pages taken first (inside the budget),
`excluded` candidates never filled, and `units`, Page sets the cosine fill takes whole or not at all.
`hooked_fill(order, room, hook)` is that fill (Selector V2's greedy loop, every Page costing one). The context is the
fill context plus `messages`, the sealed history after the initial prompt as `HistoryMessage` rows (role, token span,
call start, action text, reasoning) built by the authority from the XML transcript: a rebuilt session's re-encoded
messages first (the transcript now records their spans, `rebuilt`), then each reply since the rebuild and the
observation after it; and a per-session `state` dict for caches. The receipt and `runtime.jsonl` carry the
observation_query fields (the scores included) and the policy's record under `query.selection.hook`; the selector
config hash adds the policy's order. `protocols` limits a policy to some startup protocols (`serves`); the eight
built-in policies serve every one; the chat-text server lists only the plug-ins that serve chat-text
(`chat_text_protocol.policies()`, ledger-int-1112).

**ledger_query** (`source-overlay/page_policies/ledger_query.py`, registered in `PLUGINS`) is observation_query with
three rules from the selection study of 2026-11-11, on the Qwen XML protocol only (`mswea_xml_v1`; kv-session,
chat-text and the Gemma 4 XML grammar are refused):

1. File and edit ledger: up to `ledger_cap_percent` of the fill's room (B = capacity - mandatory - recent Pages;
   default 50, `--ledger-cap-percent`) is reserved for the call of the agent's latest edit, then the observation of
   the latest read of each file since its last edit, newest first. An item is all of its candidate Pages or none;
   one that does not fit is skipped and becomes a unit of the cosine fill (taken whole when the cosine order reaches
   it and it fits, else not at all). The rest of B is the cosine fill. When a file's latest read is truncated by the
   harness, its latest complete read since the edit is an item too.
2. Dedupe: the fill never takes a Page wholly inside the observation of a read whose every file read a later read of
   the same file covers (no edit between; a whole view covers everything, a line range the views inside it, a
   truncated output, the live tail's included, nothing), the observation of a command the agent ran again verbatim (a
   command that is not a pure read: only with no edit between the two runs), or a reply repeated verbatim.
3. Reasoning-light: the fill never takes a Page wholly inside the reasoning of a reply older than the two latest.

Commands are the reply's one `<mswea_bash_command>` block, parsed once per reply: reads (cat, nl, head, tail, sed -n,
awk line ranges, grep with file operands, with head / tail / sed -n / grep filters), edits (writes to a file that is
not an output file, sed -i, perl -i, cp / mv / rm / touch, patch, git apply / checkout ..., editors, Python bodies that
write and the scripts they wrote), diffs (git diff / show, diff); anything else is neither (environment setup such as
pip install is not an edit). The pin `{order, ledger_cap_percent, reasoning_turns: 2, dedupe: true}` is the contract's
`ledger_query` key. The record holds the room, the cap, every ledger item (kind, token span, Pages, reserved /
skipped / filled whole / already attended), the excluded Pages and why, the fill's counts, the parse counts, and the
token shares of the attended history (outside the initial prompt) that are code (file reads, diffs, edit calls),
other tool output, tool calls and reasoning, for the selection and for the plain cosine selection of the same turn.

**ledger-fixes-1116** (order `ledger_dedupe_reasoning_light_v2`, was `_v1`; merged into integrate-1116 as `a2f9e36`).
Replayed offline on job 3611601, which ran `_v1`: 594 of its 1,208 selective requests (22 of 32 sessions) select
differently (not audited); v2 has not run on a GPU yet ([docs/status.md](status.md)); what the parser still does not see is in
[docs/known-limits.md](known-limits.md) 4.1. From the verification of a review of the policy:
- Dedupe across an edit: a repeated command that is not a pure read (a test, a script run, any other output) is a
  duplicate only when no edit came between its two runs (a failing test's output before a fix is no copy of its
  output after it); a repeated pure read is a duplicate as before (the read coverage handles edits of its file).
- Python writes: `.write()` / `.writelines()` alone (a StringIO, `sys.stdout`, an astropy table) is no file write and
  no longer makes the command an edit of an unknown file (which cleared every file's read history); `open(..., 'r+')`
  and `Path(...).open('w')` are writes; a body's quoted paths resolve against the working directory, and a body
  whose quoted paths are all output files (.txt .log .out .diff .patch) is no edit.
- Truncated reads: the live tail (the newest reply's output, not in `messages`) reaches the hook as
  `FillContext.live_observation` (set by the authority on a continuing turn of the XML transcript; None otherwise,
  ignored by every other policy), so a truncated whole-file reread no longer supersedes an earlier complete read;
  when a file's latest read is truncated, its latest complete read since the last edit is a ledger item too.
- Files: resolved paths match only themselves (a repository file and its site-packages copy, or two packages'
  `__init__.py`, are different files); a path read after `cd $VAR` (unknown directory) matches by suffix.
- Parser: comments; `$(( << ))` is no heredoc; `git -C <dir>`, `git rm` / `git mv`; pushd / popd and `( ... )`
  subshells keep the working directory; `python -W` / `-X` values; `sed -i` on a file without a suffix; awk strict
  comparisons (NR > a: a + 1) and field output (a filtered read); `head -c` / `tail -c` (filtered reads);
  `sed -En` / `-nr`; `nl -n` values; `grep -ePAT`.
- Reasoning: `selection_contract.call_start` returns the reply's end for a thinking-on reply without `</think>` (cut
  while thinking: the whole reply is reasoning; before, all of it after the header was the call). This also changes
  `recent_tools` (such a reply's Pages are REASONING, not CALL) on thinking-on streams.

The launcher (same branch): `serve.py` refuses a release of this package (its `release_id`) whose installed source
files differ from the checkout's `INSTALL_RELEASE.json` (the launcher imports the checkout's registry); reinstall from
the checkout or run `serve.py` from the release's commit. The harnesses install from the checkout they run.

Not done: the goal_query variant (the hook with the Goal vector; marker-pages-1114 adds the Goal-vector hook, section
9, but `ledger_query` itself stays on the observation vector), the Gemma 4 grammar, and the harnesses' readers
(section 7: `ledger_query` is not in the `POLICIES=(...)` lines the harnesses read).

## 9. Marker pages: goal_query_markers and observation_query_markers (marker-pages-1114)

marker-pages-1114 (on ledger-int-1112 `110632b`; verified offline by a replay of recorded SWE-bench requests; its
GPU runs are in [docs/status.md](status.md)). Some token-level selection methods keep every earlier occurrence of a few
protocol markers (`</tool_response>`, `</think>`, `<|im_end|>`, `<tool_response>`) on top of their own token ranking
under a fixed token budget. The runtime restores whole 16-token Pages (FlashAttention's smallest kernel block in vLLM
0.24), so these two policies keep the Pages that hold a marker: the page approximation of section 6's option 1, inside
the budget.

**What the hook gained (for every query plug-in; the hook's contract is unchanged).**

- `FillContext.tokens`: the session's token ids, the engine transcript's own `stream` (an immutable tuple, shared, not
  copied). Positions [0, n·P) are the candidate catalog; the tuple ends at the turn's stable end, where the newest
  observation starts; between rebuilds it only grows. `FillContext.encode`: the server tokenizer's text-to-ids (no
  special tokens added, special-token text parsed), the callable the transcript encodes with.
- A query plug-in may rank by the Goal vector: `query_vector = 'goal'` with `goal_text` and `goal_engine`, exactly as
  `goal_query`. Everything the registry switches by those properties applies to it as to `goal_query`: the Goal
  forward and the Goal cache in `plan_turn`, the Goal rewriter (`--goal-rewriter-model` or `--no-goal-rewriter`
  required; `--goal-rewriter-batch`, `--goal-rewriter-engine vllm`, `--goal-overlap`), goals.jsonl's texts and the
  capacity gate's Goal transient. `register` refuses a Goal plug-in without the Goal engine and text, or with a
  `rewriter_unused` text.
- `PagePolicy.startup(pin, encode)`: the query authority calls it once per server, before any request, with the
  contract's pin and the XML transcript's `encode` (None without a transcript); a plug-in refuses its pin there by
  raising. Default: nothing.
- `PLUGIN_PIN_OPTIONS` adds `marker_set` and `marker_cap_percent` (`serve.py --marker-set`, `--marker-cap-percent`).

**The two policies** (`source-overlay/page_policies/marker_pages.py`, two `PLUGINS` entries) are `goal_query_markers`
(the Goal vector's cosine; the Goal rewriter as for `goal_query`) and `observation_query_markers` (the observation
vector's). On every active turn, after Selector V2's partition (the mandatory Pages of the initial prompt, the recent
safety Pages, the room B = capacity - mandatory - recent):

1. The markers are found in the sealed history [0, n·P). A marker is a token sequence; its first token is the marker
   token, whose Page is protected, and the tokens after it only qualify the occurrence (`<|im_start|>user`: an
   `<|im_start|>` that opens a user message, i.e. an observation of the Mini-SWE-Agent XML protocol, the analog of a
   tool response's start). Markers inside the mandatory and recent Pages are attended anyway and only counted.
2. Every candidate Page that holds a marker is protected, newest first, up to `marker_cap_percent` of B (no cap: all
   of B; never more). The cap drops the oldest marker Pages and the record counts them.
3. The rest of B is the base policy's cosine fill (`policy_registry.hooked_fill`); it may still take a dropped marker
   Page by its score.

| `--marker-set` | Markers |
| --- | --- |
| `xml_markers` (default) | `</think>`, `<\|im_end\|>`, `<\|im_start\|>user` (reasoning end, message end, observation start) |
| `ends` | `</think>`, `<\|im_end\|>` |
| `message_ends` | `<\|im_end\|>` |
| `reasoning_ends` | `</think>` |
| `tool_markers` | `</tool_response>`, `</think>`, `<\|im_end\|>`, `<tool_response>` (the four markers of a native tool-call protocol; on `mswea_xml_v1` the tool-response tags never occur) |
| a list, e.g. `"</think>,<\|im_end\|>"` | up to 8 distinct markers, each a token in angle brackets, optionally followed by a word |

At startup the server checks every marker against its tokenizer: 1 to 8 tokens, encoded the same after other text
and before a newline, the sequences distinct (with the pinned Qwen3.6 tokenizer every named set passes:
`</think>` 248069, `<|im_end|>` 248046, `<|im_start|>user` 248045 846, `</tool_response>` 248067, `<tool_response>`
248066). The scan is incremental: each turn scans only the tokens added since the last one (the session's hook state,
dropped on a rebuild or a close).

The pin `{order: marker_pages_newest_first_v1, marker_set, markers, marker_cap_percent}` is the contract key of the
policy (`marker_cap_percent` null: no cap). The receipt and `runtime.jsonl` carry the base policy's fields (the scores,
the Goal block) and the record `query.selection.hook`: the pin, `room_pages`, `cap_pages`, `occurrences` (per marker),
`marker_tokens` (distinct marker positions in the sealed history), `fixed_tokens` (those in the mandatory and recent
Pages), `marker_pages` (candidate Pages holding one), `protected_pages`, `dropped_pages`, `dropped_tokens`,
`dropped_filled` (dropped Pages the cosine fill took anyway), `kept_tokens` (markers on selected Pages),
`budget_share` (protected Pages / capacity), `room_share`, `new_pages` (protected Pages the plain cosine fill would not
have taken), `displaced_pages` (plain cosine Pages left out), `fill` (hooked_fill's counts), `scanned_tokens` and
`plain_jaccard`.

Refused: the kv-session and chat-text protocols and the Gemma 4 profile (`protocols`: `mswea_xml_v1` only; the
launcher and the startup config), a cap outside 1-100, an unknown marker set or a malformed list (the launcher), a
marker the tokenizer does not encode as above (the server's startup), `--marker-set` / `--marker-cap-percent` with
any other policy, and `--token-gather` with either marker policy (option B, token-gather-1114, serves the built-in
`goal_query` and `observation_query` only; the launcher and the contract refuse it, integrate-1114). A marker plug-in
on a non-query base (a fill policy of kind `window`, `rule` or `random`) cannot be registered: the query hook belongs
to query plug-ins.

**Offline replay (UNAUDITED).** The implemented code over the recorded requests of two SWE-bench runs at an 8,192-token
budget (512 Pages of 16 tokens), with each request's own partition, cosine order and token stream, and the pinned
Qwen3.6 tokenizer: job 3539260 (`observation_query`, thinking on; 1,198 selective requests) with
`observation_query_markers`, and job 3500565 (`goal_query` with the vLLM Goal rewriter, thinking off; 1,677) with
`goal_query_markers`. Without the hook the replay reproduces every recorded selection; the incremental scan equals a
full scan on every request. Default set `xml_markers`; Pages are shares of the 512-Page budget:

| | 3539260 (observation) | 3500565 (Goal) |
| --- | --- | --- |
| marker tokens outside the mandatory and recent Pages, mean | 129 | 180 |
| candidate Pages holding a marker: mean (median, p90, max) | 97 (80, 196, 345) | 117 (103, 223, 304) |
| their budget share: mean / p90 / max | 18.9 / 38.3 / 67.4 % | 22.9 / 43.6 / 59.4 % |
| requests whose marker Pages exceed the room (no cap: the oldest dropped) | 11 | 33 |
| no cap: markers on protected Pages; cosine Pages displaced, mean (budget share) | 99.8 %; 51 (9.9 %) | 99.4 %; 98 (19.1 %) |
| cap 50: protected Pages' budget share mean / max; markers on protected Pages | 16.5 / 34.4 %; 86.8 % | 19.7 / 39.8 %; 85.9 % |
| cap 25: protected Pages' budget share mean / max; markers on protected Pages | 12.1 / 19.7 %; 63.2 % | 13.4 / 20.1 %; 58.5 % |
| markers in the selection: base policy / no cap / cap 50 / cap 25 | 52.7 / 99.9 / 90.3 / 76.5 % | 20.4 / 99.5 / 86.9 / 62.8 % |
| page-set Jaccard with the base selection, mean (no cap) | 0.77 | 0.61 |
| hook + fill CPU time per request: median / p95 / max | 0.22 / 0.44 / 4.75 ms | 0.24 / 0.45 / 1.20 ms |

Each marker brings its Page's 15 neighbours: 1.3-1.5 markers per Page, so the page cost is about 12 times the token
cost (129-180 tokens are 1.6-2.2 % of the budget). The replay holds the agent fixed; only a GPU run shows the effect on
the agent. Job 3576565 (`goal_query`, thinking on) is not replayed: its prompts are not in the local records.

**Parity.** `tools/policy_parity.py` now covers the nine policies before this branch (the eight built-in ones and
`ledger_query` with its pin and `--ledger-cap-percent`): against `110632b`, 12,257 of 12,259 outcomes are
byte-identical; the other two (`serve.py --help` and the usage error for an unknown policy) differ only by the two new
`--policy` choices and the two new options.

**Not done.** Separators (the newest 1,024 separators span about 310-330 Pages, more than the room on 36-54 %
of the requests); per-token Goal scores (one vector per Page); the Gemma 4 profile (scope: Qwen3.6-27B first); the
harnesses' readers (section 7: the new policies are not in the `POLICIES=(...)` lines the harnesses read).
