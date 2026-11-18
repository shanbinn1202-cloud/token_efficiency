# Gemma 4 thinking "the same as Qwen" (design, 2026-11-09)

English | [简体中文](DESIGN.zh-CN.md)

Design of thinking on for the Gemma 4 31B IT profile, decided by the maintainer on 2026-11-09: the full scope, Tier 1
(keep: `--enable-thinking` on `mswea-xml` and `chat-text`, `--kv-session-thinking on`) and Tier 2
(`--thinking-history strip`, form E, on `chat-text`). As built: the work was done on two branches, `gemma-thinking`
(this note; Tier 1 and the transcript side of strip) and `gemma-thinking-hold` (the held cut window of section 4.4),
integrated on `gemma-thinking-int` (`3219296`) on 2026-11-10, and is on `main` since integrate-1112 (`4ebff97`,
2026-11-15): `--thinking-history strip` on Gemma is wired end to end (section 4.4). Its GPU checks (section 7.2) and
their verdicts are in [docs/status.md](../status.md) and [History](../history.md#history); the limit that holds for
Gemma's keep mode is [docs/known-limits.md](../known-limits.md) 1.2. The two "As built" paragraphs below record the
branches and their CPU tests on 2026-11-10; the rest of this page is the design record.

**As built on `gemma-thinking` (2026-11-10, CPU tests).** Tier 1: `--enable-thinking` on Gemma's XML protocol and chat-text and
`--kv-session-thinking on` (the launcher, the pins, the startup checks against the pinned template, the transcripts,
the routes, MarkerStop and its async prediction), `tests/test_gemma_thinking_on.py`. Tier 2, the transcript side
of strip: the Gemma reply piece and head (K = |H|), its template check, the commit at the cut and the re-prefill,
the `--async-scheduling` refusal, `tests/test_gemma_thinking_cut.py`. With thinking off, and on every Qwen path, the
launches, records, transcripts and responses are those of `ba4d1ba` (identity probes over 729 launches and 35
protocol scripts: 0 differences). On Gemma the keep and strip `chat_text` records are the same (header and template
arguments); the `thinking` record's `history` tells them apart.

**As built on `gemma-thinking-int` (2026-11-10, CPU tests).** The held cut window (option G): `remote_ra_live_v2/window_hold.py`
with its engine, worker and capacity parts (section 4.4). `thinking_cut.attach_engine` and `attach_worker` hand the
window layout to it, the window worker takes and echoes the strip record, the startup check requires hold slots of a
Gemma strip config, and `serve.py --window-hold-slots H` sizes them (left out: the most that fit). Tests:
`tests/test_gemma_hold.py` and `tests/test_gemma_strip_e2e.py` (section 7.1). Identity probes of the integrated
tree, all with 0 differences: against `ba4d1ba`, the 729 items of `gemma-thinking`'s launcher probe (Qwen with
thinking off, keep and strip; Gemma with thinking off; the Goal rewriter's commands; the startup evidence), its 35
protocol scripts, the 10 engine runs of the window layout (122 files) and the 677 items of
`gemma-thinking-hold`'s launcher probe; against `gemma-thinking`, 278 items of Gemma's keep mode and
`--kv-session-thinking on` (launches and transcript steps). Without `--thinking-history strip` no hold slot is
reserved and none of the hold's code runs.

References below: `TPL` is the pinned Gemma `chat_template.jinja` (SHA-256 `ae53464b...`, pinned in
`remote_ra_live_v2/model_profile.py`; it is not in the tree, tests read it through `REMOTE_RA_GEMMA4_METADATA`),
`TCFG` its `tokenizer_config.json`. Line numbers of the runtime are those of `main` `ba4d1ba`.

## 0. Decisions (2026-11-09)

1. **Scope.** Tier 1 (keep on `mswea-xml` and `chat-text`, `--kv-session-thinking on`) and Tier 2
   (`--thinking-history strip`, form E, on `chat-text`) for the Gemma 4 profile.
2. **Terminal-Bench history on Gemma: strip**, the stock / Vanilla input (section 1.4).
3. **The window at the cut: option G**, a GPU hold with a shared reserve of H hold slots and a decode cap
   (section 4.3; in effect a cap on the strip turns in flight). H = 0 (no strip) leaves everything byte-identical; the launcher sizes or refuses H against the
   pool with the selective rule. The host copy (option H) and 32 GiB of KV are not built.
4. **Reply split:** vLLM 0.24.0's `gemma4` reasoning parser decides (section 2). It agrees with "no channel =
   content only; an unclosed channel = all reasoning" and adds rules for a channel that is not leading, a close
   without an open and a `thought` prefix without a channel; the runtime follows vLLM there.
5. **Baselines:** every Gemma SWE-bench and MemoryAgentBench baseline config of the collaborator's repository sets
   `enable_thinking: true` (section 5).
6. **MemoryAgentBench:** the runtime side (`--kv-session-thinking on` for Gemma) is in Tier 1. The MAB client
   (`examples/memoryagentbench/`) is other people's code and keeps its thinking-off Gemma tail.
7. **Refusals in v1:** `--async-scheduling` with Gemma strip; strip on `mswea-xml` and `kv-session` (as on Qwen).
8. **Order:** CPU development on the branches now; the GPU checks after the Gemma async run is verified; `main`
   only after GPU validation and the maintainer's approval.
9. **Invariants:** the Qwen paths stay byte-identical, and so does Gemma with thinking off (identity probes).

## 1. Gemma 4's own thinking format (pinned template and tokenizer)

### 1.1 Tokens

| Token | Id | Special |
| --- | --- | --- |
| `<\|think\|>` | 98 | yes (`TCFG` `think_token`) |
| `<\|channel>` / `<channel\|>` | 100 / 101 | yes |
| `<\|turn>` / `<turn\|>` | 105 / 106 | yes |
| `<\|tool_call>` / `<tool_call\|>`, `<\|tool_response>` / `<tool_response\|>` | 48 / 49, 50 / 51 | yes |
| `thought` | 45518 | no (plain BPE; `thought\n` is `[45518, 107]`, `thought\n\n` is `[45518, 108]`) |

The stop ids are {0, 1, 50, 106}; `<channel|>` does not stop a reply.

### 1.2 What `enable_thinking` changes (`TPL:186-196`, `TPL:381-390`)

- **The default is off** (`enable_thinking | default(false)`); Qwen's template defaults to on. A stock Gemma control
  asks for thinking explicitly (`chat_template_kwargs: {"enable_thinking": true}`).
- **The mandatory prefix changes.** On, the first system turn starts with `<|think|>\n`. Without a system message
  (Terminus 2 sends none) the template adds a whole system turn, `<bos><|turn>system\n<|think|>\n<turn|>\n`: 7 more
  tokens `[105, 9731, 107, 98, 107, 106, 107]`. With a system message it adds 2 (`<|think|>\n`). Qwen's flag changes
  only the generation prompt.
- **Generation header after a user turn:** off `<|turn>model\n<|channel>thought\n<channel|>` (7 tokens, the empty
  thought channel of today's thinking-off servers); on **`<|turn>model\n`, 3 tokens** `[105, 4368, 107]`: the model
  opens the channel itself.
- **After tool responses:** off no header (the model turn goes on, today's kv-session `header_tokens` 0); on
  **`<|channel>thought\n`, 3 tokens** `[100, 45518, 107]`: the template opens the thought channel after every tool
  response.

### 1.3 What the model emits (`TCFG` `response_schema`, `response_template`)

A thinking reply is `<|channel>thought\n` R `<channel|>` C `<turn|>`; the channel is optional. Two consequences,
both unlike Qwen:

- **The markers are special tokens.** Decoding with special tokens skipped (what chat-text and XML do) loses them:
  `<|channel>thought\nLet me look.<channel|>{"a": 1}` decodes to `thought\nLet me look.{"a": 1}`. Qwen's `</think>`
  is not a special token. Gemma must split **on ids** (section 2).
- **History renders reasoning as `<|channel>thought\n` + R + `\n<channel|>`** (`TPL:242`), with a newline before the
  close that the model's raw output may not have. A client that re-renders R (any `preserve_thinking` render) can
  differ from the raw ids by that `\n` token; how often is a GPU measurement (section 7.2).

### 1.4 What the template does with earlier thoughts (`TPL:156-166`, `218-243`, `323-349`)

A model message's reasoning renders only after the last user message, or with `preserve_thinking` on a message
with `tool_calls`; its content always goes through `strip_thinking` (every `<|channel>...<channel|>` removed, then
trimmed).

| Conversation shape | Default (`enable_thinking=True`) | `preserve_thinking=True` |
| --- | --- | --- |
| chat-text / our XML: observations are **user** messages | every past reply `<\|turn>model\n{C}<turn\|>\n`, **R dropped** | the same (no `tool_calls`) |
| tool loop inside one user turn (MAB's `read_next`, a tool-calling SWE harness) | R kept after the last user message | the same |
| a new user message after the tool loop (MAB's question) | every earlier R dropped | R kept |

**For text turns Gemma's natural mode is strip**, and no template argument keeps R there. A strip stream equals the
stock thinking-on render exactly; today's thinking-off stream keeps the empty channel, 4 tokens per past reply more
than stock.

## 2. The reply split (decision 4: vLLM 0.24.0's `gemma4` parser)

Read at the `v0.24.0` tag: the registry names `gemma4` -> `gemma4_engine_reasoning_parser.Gemma4ParserReasoningAdapter`,
an adapter of `vllm/parser/gemma4.py` `Gemma4Parser` on the generic parser engine (`vllm/parser/engine/`). Our
clients are non-streaming; the non-streaming path is `extract_reasoning(model_output)`:

1. With thinking on the request decodes **with** special tokens (`skip_special_tokens=False`), so the text holds the
   markers.
2. The engine starts in CONTENT on every call. `<|channel>` in CONTENT -> REASONING; `<channel|>` in REASONING ->
   CONTENT; `<channel|>` in CONTENT is swallowed; a terminal without a transition (a second `<|channel>` inside
   reasoning) stays as text of the state (an open point, below). A reasoning still open at the end stays reasoning.
3. **Injection** (first feed only): a text without `<|channel>` that holds `<channel|>`, starts with `thought\n` or
   equals `thought` gets a virtual `<|channel>` in front.
4. Reasoning = the reasoning text `rstrip()`ed (`strip_trailing_reasoning_whitespace`, default on), then a leading
   `thought\n` removed (exactly `thought`: no reasoning); empty -> `None`. Content = the content text; empty -> `None`.

The runtime applies the same rules on the generated ids (`remote_ra_live_v2/thinking.py`, `Gemma4Split`): the stop
id at the end is dropped (vLLM's text excludes it), the ids are cut at 100 / 101 into runs, each run is decoded
(special tokens skipped, as today's replies), and the state machine above assigns the runs. The response carries
`content` = C and the reasoning in `reasoning_content` and `reasoning`, as on Qwen.

| Case | vLLM | Runtime | Decision 4's wording |
| --- | --- | --- | --- |
| no channel, no `thought\n` prefix | content only | the same | the same |
| leading channel, closed | reasoning (prefix and trailing whitespace stripped), content after | the same | the same |
| channel never closed | all reasoning, content `None` | all reasoning, content `""` | content `""` |
| channel after some content | content before + after, the channel as reasoning | the same | (not covered) |
| `<channel\|>` without `<\|channel>` | reasoning before it | the same | content only |
| no channel, text starts with `thought\n` | reasoning | the same | content only |

Remaining differences: content and reasoning are `""` where vLLM sends `null` (Qwen's rule, "content is never
null"); the runtime matches the markers on ids, vLLM on text (a plain-text spelling of `<|channel>` would also split
in vLLM; the model writes the markers as special tokens); other special tokens inside a reply are skipped in our text
(as today), and `<|tool_call>` inside reasoning (vLLM: end of reasoning) does not occur on these protocols, which
serve no tools.

**Open point: a second `<|channel>` inside reasoning** (the review of 11-10, finding 4). The runtime keeps a nested
open marker as text of the reasoning (`<|channel>thought\nx<|channel>y<channel|>z` gives the reasoning
`x<|channel>y`; `thinking.py`, `Gemma4Split`). Two summarising reads of `vllm/parser/gemma4.py` at the `v0.24.0` tag
(WebFetch, the reviewer's) show a transition (REASONING, `THINK_START`) -> REASONING with no output and a comment about
swallowing it, which would give `xy`. Neither read is byte-exact, and the tests' model of the parser
(`test_gemma_thinking_on.vllm_gemma4`) is this note's own reading, so its random cases cannot settle it. Only the
reasoning text (and with it the reasoning's token count) would differ: the content, and so every strip prompt, is the same either way.
Open until vLLM's parser is read byte for byte (or GG1's check 6 meets such a reply); the runtime is unchanged.

**MarkerStop (XML).** The action marker counts only in the content of the reply so far (the split above applied to
the output ids). The incremental argument of `MarkerStop` holds with one addition: a channel id (100 or 101) can move
earlier text between reasoning and content, so such a token decodes the whole output, as a token holding `>` does.

## 3. The three switches

### 3.1 Mapping

| Qwen | Gemma equivalent | Template reference on Gemma |
| --- | --- | --- |
| kv-session `--kv-session-thinking on`: the client renders, the server appends raw ids, R kept | the same, with two pinned headers: after a user turn `<\|turn>model\n`, after tool responses `<\|channel>thought\n` | `enable_thinking=True, preserve_thinking=True` (R kept for tool-call messages) |
| `--enable-thinking` keep (`mswea-xml`, `chat-text`): the stream keeps R's ids; append only when R comes back, else rebuild | the same rules; header `<\|turn>model\n`; split on ids; MarkerStop after `<channel\|>` | none for text turns: the rebuild renders `<\|turn>model\n<\|channel>thought\n` R `<channel\|>` C `<turn\|>\n` |
| `--thinking-history strip` (`chat-text`): form E, commit at X, re-prefill the returned C | the same transcript; the engine holds the sliding window at X (section 4) | the default template: every past reply `<\|turn>model\n{strip_thinking(C)}<turn\|>\n` |

### 3.2 Shared with Qwen (no change)

The reasoning read from the raw request body, the response fields, the append rule (content and reasoning compared
after `strip()`), the history evidence names, the routes, strip's `plan` and `_commit_cut`, the engine's global-pool
commit at X, the evidence of the cut, the Goal's previous reply = C, and the reply bound: `--enable-thinking`
defaults `--max-tokens` to 4096 for any profile (the reasoning counts inside it, as in the baselines).

### 3.3 Gemma specifics

- **Pins.** The `thinking` record is profile-aware: Gemma's names the header `<|turn>model\n`, the template
  arguments `{enable_thinking: true}` (no `preserve_thinking`: it changes nothing for text turns), the channel
  markers, the `thought\n` prefix and the split rule. Qwen's record is byte-identical.
- **Initial render.** `enable_thinking=True`: the `<|think|>` system turn (section 1.2), checked at startup.
- **chat-text keep.** A past reply is `<|turn>model\n` + (`<|channel>thought\n` R `<channel|>` when the message
  carries a reasoning) + `strip_thinking(C)` + `<turn|>\n`. A client that drops R rebuilds every turn into exactly the
  stock render (strip's history); one that sends R back appends.
- **XML keep.** A rebuild re-encodes every past reply from its content (`<|turn>model\n` + `strip_thinking(C)` +
  `<turn|>\n`, R not re-rendered, as on Qwen); that is also the template's own render of our XML shape.
- **kv-session.** Two headers pinned at startup from the template; a request ends with one of them (the tool header
  after `<tool_response|>`); the thinking-off path with `header_tokens` 0 stays for thinking off only.
- **strip pieces.** The reply head `<|turn>model\n` **is** the whole header: K = |H| = 3 (Qwen 3 < 5), so the
  strip head check allows K = |H| on Gemma. `strip_content` is the template's `strip_thinking`.

## 4. Strip (form E) in the window layout

### 4.1 Geometry (chat-text, thinking on)

Header H = `<|turn>model\n`, 3 tokens. Turn t: prompt N_t, X_t = N_t - 3. The next prompt re-renders the reply as
`<|turn>model\n` C `<turn|>\n`; its first 3 tokens are the old header, so the stable end is X + 3 and the turn
geometry is Qwen's (C = X < o = X + 3 < e < N). Phase 1 re-prefills `<|turn>model\n` + C + `<turn|>\n` + the user
piece; Gemma's phase 1 already runs in chunks when it exceeds one native forward.

### 4.2 Why the cut needs a hold

Gemma has no GDN state. What crosses turns is the sliding window: the session pins the blocks of [C' - 1023, C') in
every sliding group. At the phase-1 end the window at e = X exists, and the handover gives it to phase 2. But phase 2
does not keep it: vLLM's sliding-window manager frees each block whose tokens all lie before the next query's
window, so after about 16 decoded tokens the oldest block of the window at X starts to go, and after about 1,024 the
whole window is gone. Recomputing the window at the next turn needs a full 60-layer forward of [X - 1023, X), which
also writes the global KV of sealed Pages; it is rejected as on Qwen.

### 4.3 Option G: hold the window on the GPU

One extra reference on the handed-over window blocks (a continuing turn) or on the window at X once the prefill
passes X (local turn 0); `pin_after(X)` from the held references; phase 2's own blocks free as today. Cost: up to one
more window slot per decoding strip session, 65 blocks x 5 sliding groups x 2.5 MiB = **812.5 MiB at P = 16**,
reserved at admission (a 4,096-token reply outlives the 1,024-token window). At 25 GiB of KV only a shared reserve of
H holds fits. The table is computed by the runtime's own sizing (`kv_layout.window_pool_blocks`,
`window_capacity.selective_pages` and `most_hold_slots`; `tests/test_gemma_hold.py` `CapacityTableTests` recomputes
every cell): pool-0 Pages left beyond the launcher's selective rule with an 8,192-token selection (negative =
refused), and the most hold slots that fit, which is the launcher's default. With hold slots the rule also counts
the reply a strip turn re-prefills (section 4.4, "The launcher"), 256 Pages at 4,096-token replies:

| Server (P = 16, 4,096-token replies, 25 GiB) | H = 0 (no strip) | H = 1 | H = 2 | H = 3 | H = 4 | H = 5 | Most H |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TB `observation_query`, K = 20, 262,144 tokens, 4,096-token forwards | +2,798 | +1,899 | +1,255 | **+612** | -32 | -675 | 3 |
| TB `goal_query`, K = 20 | +1,531 | **+632** | -12 | -655 | -1,299 | -1,942 | 1 |
| SWE `observation_query`, 16, 131,072, 8,192-token forwards | +2,330 | +1,430 | +787 | **+143** | -500 | -1,144 | 3 |
| SWE `goal_query`, 16 | +1,063 | **+163** | -480 | -1,124 | -1,767 | -2,411 | 1 |

One hold slot is 325 block ids of pool 1 (65 per sliding group), 812.5 MiB, the room of 650 global Pages: on the
first line pool 1 grows from 19.00 GiB (H = 0) to 21.38 GiB (H = 3) and pool 0 shrinks from 4,918 to 2,968 Pages.
The two SWE lines only show the sizing: strip serves `chat-text` (section 4.5). The study of 11-09 listed H = 2 for
the third line; H = 3 fits there too. Its rule had no term for the re-prefilled reply (the review of 11-10): by
it the first two lines took 4 and 2 hold slots, which the term leaves short by 32 and 12 Pages. A session slot and a
hold slot take the same room, so one session fewer buys one hold slot back (`goal_query` with 19 sessions: H = 2,
+633 Pages). With 24 sessions on the first line (the most that fit without a hold slot, +220 Pages) not even one hold
slot fits (-679), and the launcher refuses a strip server there.

**What H caps.** A strip turn takes its hold slot at its admission and gives it back at its commit (or when it
fails): the gate (`window_capacity.hold_wait`) counts every admission in flight or parked (lifecycle states
`rendered`, `reserving` and `accepted`) and every hold in the engine. So H caps the strip turns in flight, not only
the replies that decode: phase 1, a `goal_query` admission parked at its Goal and a cold prefill (a first turn, a
rebuild) each take a slot. The gate waits, it never preempts. With `--admission-overlap K` at most min(K, H)
admissions are in flight, and a Goal batch (`--goal-rewriter-batch N`, which sends the calls queued at that moment,
up to N) holds at most H calls: at H = 1, the most that fit on the Terminal-Bench `goal_query` line, the server
runs one turn at a time from its admission to its commit, and every Goal batch has one call.

**A smaller H can serve more.** The most that fit (the launcher's default) leaves pool 0 at the selective rule's
minimum, which holds one worst-case turn (its own pins included) next to the other sessions' tails, not H of
them; idle sessions' Pages are released for it. Pool 0 of the Terminal-Bench lines at 25 GiB with K = 20, from the
code (`window_capacity.pool_zero_blocks` and `selective_pages`), global Pages and in brackets the rule's figure:

| Terminal-Bench line, K = 20 | H = 0 (no strip) | H = 1 | H = 2 | H = 3 |
| --- | --- | --- | --- | --- |
| `observation_query` | 4,918 (2,119) | 4,268 (2,368) | 3,618 (2,362) | 2,968 (2,355) |
| `goal_query` | 3,638 (2,106) | 2,988 (2,355) | refused | refused |

Every hold slot takes 650 Pages that would otherwise hold restores, readmissions and rows of turns running
together. Once contexts pass the 8,192-token threshold a turn keeps a selection of 512 Pages resident and adds up
to one forward and one reply (513 Pages) per run, so the largest H can leave the gate waiting for global Pages with
hold slots free (`CAPACITY_WAIT: FA pool`), and a smaller H may run more turns at once. Which H serves Terminal-Bench
best is a measurement (GG2 counts the waits by cause). The host copy (option H, about 16 GiB pinned at K = 20) and
32 GiB of KV were not chosen.

### 4.4 The hold as built (`remote_ra_live_v2/window_hold.py`)

**What is held.** The sliding window at the cut X: the blocks of [X - 1023, X) (`sliding_window.kept_blocks(X)`) in
each of the five sliding groups, at most 65 blocks per group, one window slot. A strip request takes one more
reference on them and keeps it while phase 2 decodes past X; vLLM frees the request's own row as today.

| Step | Engine (`window_pins.WindowPins`, `adapter.EngineCoreAdapter`) | Worker (`window_owner`, `window_lifecycle`) |
| --- | --- | --- |
| continuing turn, at the install | `hold(..., 'install')`: one more reference on the installed head, the window phase 1 ended in (the window at e = X) | the cut window is read from the native rows of phase 2's first forward and checked against the window phase 1 ended in |
| local turn 0 | `hold(..., 'prefill')` before the first allocation at a cursor at or past X; a reply of one token that ended at the step that crossed X is pinned from its row, without a hold | the cut window from the native rows of the prefill forward that holds or ends at X; no forward has to end at X, because the window's KV is exact whatever chunk computed it |
| commit | `pin_after(X)` from the hold: its references become the session's pin; the session commits at X as on Qwen (the seal at floor(X / P), the tail with valid end X) | the drain commits the `WindowCarry` at X |
| failed admission, cancel, abort | the hold is released | the held block keys go at the request's drain |

**H, the hold slots.** `kv_layout.hold_slots` = H. Pool 1 reserves H more window slots: 1 + 5 x (S x 65 + H x 65 +
the forward's transient + the Goal transient) block ids. At most H strip requests that hold or will hold a window
are admitted at once: the capacity gate (also the ledger gate of `--admission-overlap` and the gate before an idle
release) answers `CAPACITY_WAIT: window hold slots: H of H in use`, and the turn waits; nothing is preempted. Each
strip turn's gate figures carry its hold term, min(|kept_blocks(X)|, ceil((3 + max_tokens) / P) + 1) blocks per
sliding group, a whole slot at the 4,096-token reply bound.

**The launcher.** `serve.py --window-hold-slots H` (1 to the running sessions; the Gemma 4 profile with
`--thinking-history strip` only) reserves H hold slots. H is checked against pool 0 by the server's rule (the
selective rule on the benchmark lines) and refused when the Pages left do not hold it. Left out, the launcher takes
**the most that fit** (the last column of the table in 4.3) and refuses the server when not even one fits. The
number is pinned as `kv_layout.hold_slots` in `startup.json` and in vLLM's `--additional-config`, the only vLLM
argument strip changes; the dry run and `launch.json` name it with its source (`window_hold`), and `ready.json`
carries the layout the worker echoed. The default gives the most strip turns in flight and the smallest pool 0 the
rule allows; a benchmark plan should pin H, so that a later change of the sizing cannot move it.

**The re-prefilled reply in the selective rule** (`window_capacity.strip_reprefill_pages`; the review of 11-10,
finding 8). The rule's worst turn holds two runs of one forward and one reply each: the run sealed in the last turn
(resident) and this turn's row. A strip turn's phase 1 computes the previous reply's content again before the
observation, in chunks when the two exceed one forward, so its run is up to one forward plus one reply bound of
content. The commit at the cut drops the last reply, which leaves room for one such content, but the turn holds two
(the content re-prefilled last turn, in the run it sealed, and this turn's) next to its own reply: the rule without a
term covered a strip turn only while those two contents stay within one reply bound together. With both at the
4,096-token bound the gate's own count of the turn (its pins and phase-2 need, what the idle release compares with
pool 0 before it answers `CAPACITY_EXCEEDED`) was 255 Pages above the rule on the Terminal-Bench line
(`StripRuleTests`). With hold slots the rule therefore adds ceil(max_tokens / P) Pages, 256 at 4,096-token replies,
to both of its peaks; the launcher's sizing of H, its refusals and the `request_pages` it pins for vLLM's startup
check all take it. Without hold slots (every server but a Gemma strip server) the figure is unchanged.

**Evidence.** The native file's `truncate` adds `window_hold`: `source` (`install`, `prefill` or `row`), `end` (X),
`first` and `last` (the blocks of `kept_blocks(X)`), `held_at` (the cursor when held) and `left` (how many held
blocks of the first sliding group had left the request's row by the commit). The runtime row's
`truncate.window_hold` is the same record.

**A hold whose cleanup failed** (the review of 11-10, finding 6). A cancel or abort in phase 2 whose drain fails
(the worker's `cancel_and_drain` raises) leaves the lifecycle entry `drain_pending` with its `cleanup_error`, the
session poisoned, and the hold in place: `after_drain`, which aborts the request so that the native free releases the
hold (`WindowPins.after_free`), runs only after a drain that succeeded. The error goes to whoever cancelled (the
route's Cancel; on vLLM's abort path the compat lifecycle's `free_request` raises it). Until then the hold keeps its
slot and `hold_wait` counts it: with H = 1 every other strip turn waits (`CAPACITY_WAIT: window hold slots: 1 of 1 in
use`). It does not wait for ever: `drain_pending` is expirable (`lifecycle.EXPIRABLE`), so once the admission's lease
has run out (from the admission: `LEASE_SECONDS`, 360 s; with `--admission-overlap K` 360 + 120 (K - 1) s, 2,160 s at
K = 16) the engine loop's lifecycle sweep (`compat.EngineLifecycle.tick`, which calls `EngineLifecycle.expire`)
retries the drain. A retry that succeeds aborts the request, the hold is released and the waiting turns run; one that
fails raises `lease expiry drain failed` out of the sweep, which `core.py` runs without a handler, so the engine
core stops with that error. A hold whose cleanup failed therefore takes its slot for at most its admission's lease
and never silently longer; no code change (`test_gemma_hold.py` `CleanupFailureTests` pins each step).

**Interactions (CPU fakes, section 7.1).** `--goal-overlap` (its trim frees only blocks older than the window at
e), `--admission-overlap`, `--release-idle-pages` (the windows stay on the GPU), the three decode fast paths,
`--deferred-completion-io` and a phase 1 in chunks compose with the hold. Without the strip history H = 0: no field
in the layout record, no block reserved, no gate term, and none of the hold's code runs.

### 4.5 Refusals in v1 (decision 7)

- `--async-scheduling` with Gemma strip (async C voids rows in every sliding group): refused by the launcher, the
  startup check and the window worker.
- Strip on `mswea-xml` and `kv-session` (as on Qwen).
- Gemma strip without a hold slot: `--window-hold-slots 0`, a line on which not even one fits (the launcher), a
  startup config whose `kv_layout` has no `hold_slots` (the startup check), a layout without hold slots (the engine
  and worker attachments).
- `--window-hold-slots` without `--thinking-history strip`, and on a Qwen release (its strip commit holds a GDN
  state, no window).

## 5. Benchmarks and baselines

| Benchmark | Protocol | Natural history with thinking | Gemma mode "as Qwen" |
| --- | --- | --- | --- |
| Terminal-Bench 2.0 (Terminus 2) | chat-text | strip (observations are user messages) | strip (= the stock / Vanilla input); keep possible, no stock equivalent |
| SWE-bench Verified (Mini-SWE-Agent 2.4.2 XML) | mswea-xml | strip for our XML; keep for a tool-calling harness | keep (as Qwen) |
| MemoryAgentBench (`read_next` loop) | kv-session | keep while reading; the default drops R at the question | `--kv-session-thinking on`; the template forces a thought after every tool response |

The collaborator's repository (`token_efficiency`, read at `068efec`, 2026-10-28): every Gemma config under
`experiments/gemma_full_20261017/formal/` sets `model.enable_thinking: true`, for SWE-bench
(`swebench/<method>/8k.yaml`) and MemoryAgentBench (`memagent/<method>/memagent/32k.yaml`), for all five methods
(vanilla, agentkv, intentkv, random_attention, sliding_window), with `max_tokens: 4096` and
`reflection_enable_thinking: false` for the reflection call; the Terminal-Bench configs were read as
`enable_thinking: true` on 10-30. The Gemma baselines therefore run thinking on, and our Gemma runs compare with them
only with thinking on.

## 6. Risks

1. **Capacity (strip).** 812.5 MiB per held window; at 25 GiB H <= 3 (TB obs), 1 (TB goal), 3 (SWE obs at 8,192)
   or 1 (SWE goal): the gate caps the strip turns in flight (section 4.3) and throughput drops.
2. **Model behaviour.** Gemma may skip the channel or think until the bound; the `\n` before `<channel|>` in a
   history render versus the raw output can make a re-rendering control differ by a token per reasoning block.
3. **No template reference for keep on text turns.** Gemma keep runs compare only with the group's KV-session methods
   (AgentKV, Sliding Window, Random), not with Vanilla. Strip on TB is the only stock-identical input.
4. **Prefix change.** 7 more tokens in the initial segment (2 with a system message); thinking-on runs are a separate
   configuration from the thinking-off ones.
5. **Window lineage on the GPU** with the compact rows of fast paths 2 / 3, incarnations and CUDA graphs: the main
   GPU-only risk; GG1's cut-versus-rebuild pairs check it.

## 7. Tests and GPU checks

### 7.1 CPU

- `test_gemma_thinking_on.py` (Tier 1): pins and refusals; the split (no channel, closed, unclosed, a channel inside
  C, the injection cases); MarkerStop after `<channel|>`; appends and rebuilds on both protocols; with
  `REMOTE_RA_GEMMA4_METADATA`, the pinned template's initial segment with `<|think|>` and the headers; a
  MemoryAgentBench-shaped kv-session stream equal to the template's render with `enable_thinking=True,
  preserve_thinking=True`; end to end, Terminus 2 sessions on the Gemma chat-text server and a kv-session read loop
  through the route server; identity probes (Qwen and Gemma thinking-off records, headers and streams unchanged).
- `test_gemma_thinking_cut.py` (Tier 2, transcript side): T1, the reply piece equals the stock render for random
  contents (inline channels included), head = header; T2, Terminal-Bench-shaped sessions through Gemma's turns with
  synthetic reasoning (Terminus 2's client: plain episodes, the summary hand-off, a reply cut at its bound,
  context refusals) and Terminus 2.0.0's exact prompts with the summary hand-off, every prompt equal to the stock
  render (no 4-token difference), with the pinned tokenizer and template when `REMOTE_RA_GEMMA4_METADATA` names
  them; the 178 recorded Terminal-Bench trials are private, so their replay is opt-in only
  (`REMOTE_RA_TB_ATIF_DIR`, counts only). The pins: a Gemma strip config needs hold slots, the launcher's
  `--window-hold-slots`, the dry run's and `ready.json`'s record of H.
- `test_gemma_hold.py` (the held cut window, T3 / T4): the layout's `hold_slots` and its pin; the capacity table of
  section 4.3 recomputed; the launcher's sizing and refusals; `WindowPins` over vLLM's sliding-window manager at
  W = 1,024 (holds at the install and at local turn 0, replies of fewer than 16, about 1,000 and more than 1,024
  tokens, cancel and discard); end to end on the window layout's engine and worker behind the Gemma chat-text server,
  with the runtime's Gemma strip transcripts over a stand-in tokenizer whose header is the pinned 3 tokens: the
  commit at X with the held window, the turn after a cut, local turn 0, a cancel in phase 2, a rebuild after a cut,
  the gate with one hold slot, a full pool; the three decode fast paths, `--deferred-completion-io`,
  `--admission-overlap`, `--release-idle-pages`; mutations of the hold; nothing runs with H = 0.
- `test_gemma_strip_e2e.py` (the integration): replies shorter than a block, just under a window and longer than a
  window, a reply without a channel and one cut inside its channel; Terminus 2 sessions with the summary hand-off,
  every served prompt equal to the stock thinking-on render; a re-prefill longer than one native forward (phase 1
  in chunks); `--goal-overlap` with the hold on the goal_query scenario (with `--admission-overlap`,
  `--release-idle-pages` and the fast paths), every session equal to the same server without `--goal-overlap`; the
  strip record between the engine and the window worker; mutations of the wiring.

### 7.2 GPU (ghx4 batch, one GH200; GG1 ran, GG2 has not run; results in docs/status.md)

- **GG1, token alignment** (as THG1): one session, `full`, eager and `--cuda-graphs full`, a strip server and a keep
  server against stock vLLM 0.24.0 with `--reasoning-parser gemma4` and `enable_thinking: true`: every prompt equal
  to stock's (strip exactly), replies against stock's own spread, the turn after a cut against a forced rebuild, cold
  probes ((X - 1023) mod 16 at 0, 1 and 15; the header straddling a forward boundary; replies of fewer than 16, about
  1,000 and more than 1,024 tokens; an unclosed channel; a reply without a channel), a kv-session keep session, and
  the `\n` before `<channel|>` measured. About 2-2.5 GPU-h.
- **GG2, Terminal-Bench dry run** (as strip25): `gstripobs262`, `gstripgoal262` and a stock thinking-on control,
  16 tasks, at 262,144 tokens, 25 GiB, K = 20, with H = 3 for `observation_query` and H = 1 for `goal_query` (the
  most that fit, section 4.3). About 2-2.5 GPU-h.
- **Smokes (Tier 1):** a Gemma SWE keep smoke (about 0.9 GPU-h) and, once the client supports it, a Gemma MAB
  thinking-on smoke (about 0.75 GPU-h).
