# Qwen3.6-27B thinking ON: analysis (2026-10-30, read-only, branch thinking-on @ cf2b28d)

English | [简体中文](thinking-on-analysis.zh-CN.md)

> **Unreviewed.** An agent's analysis of 2026-10-30, kept as it was written; nobody has reviewed it line by line, and its numbers are not audited. **Frozen on 2026-11-16** (at `ccee5c6`): moved here from `docs/thinking-on/ANALYSIS.md` by docs-reorg-1116, unchanged except for this note and the language link; not updated. What was built from it: sections 6.8 and 6.9 of [docs/running.md](../running.md#6-harness-interface), [docs/thinking-cut/DESIGN.md](../thinking-cut/DESIGN.md) and [docs/thinking-gemma/DESIGN.md](../thinking-gemma/DESIGN.md); its current limits are in [docs/known-limits.md](../known-limits.md).

Agent report, saved verbatim in substance by Claude (not reviewed line by line yet). Rendering checked with the pinned
tokenizer and the Qwen template (sha256 e84f32a2...), by a local check script that is not in this repository. Not
verified locally at first: the clients' behaviour (Mini-SWE-Agent 2.4.2 and its litellm; sources not available here).
The addendum's stub probe then ran Mini-SWE-Agent 2.4.2 and Terminus 2 (Harbor 0.21.0, litellm 1.97.0).

## Status (2026-11-08)

- **Built, on main:** stage 1, `--enable-thinking` (`94888e1`; mswea-xml and chat-text on Qwen), and option (a) as
  `--thinking-history strip` (`8ca8511`; chat-text on Qwen; [docs/thinking-cut/DESIGN.md](../thinking-cut/DESIGN.md)).
- **GPU:** functional GPU checks of `--enable-thinking --thinking-history strip` passed (jobs 3493649, 3500646,
  3499759), all at runtime `8ca8511`: job 3493649 THG1_PASS_VERIFIED (G1), job 3500646 THG1B_PASS_VERIFIED (G1b) and
  job 3499759 TB16_PASS_VERIFIED (a Terminal-Bench dry run). No full benchmark yet.
- **Stage 1's keep mode** (`--enable-thinking` without `--thinking-history`) ran on the GPU only as job 3500604's
  kref / kovl pair (mswea-xml, runtime `c07d42a`): a `--goal-overlap` check, CMG1_PASS_VERIFIED, the pair EXACT (the
  Goal equal on 18 of 18 lockstep turns). That compares `--goal-overlap` on and off with keep mode on both sides; keep mode is not validated on
  its own. SWE-bench runs with it are queued (jobs 3539242, 3539260, 3540110 and 3549065).
  Since then (2026-11-15): jobs 3539242 (`recent_fill`) and 3539260 (`observation_query`) ran, both at `60c5d21` and
  SWE32_PASS_VERIFIED; jobs 3540110 and 3549065 were cancelled before they started (`akv_rule` with thinking on ran
  in job 3577833, SWE32_PARTIAL_VERIFIED).
- **Settled:** the server splits the reasoning from the content when thinking is on (content C, `reasoning_content`
  and `reasoning` R), and MarkerStop ignores markers inside R (stage 1). Option (a), cut and re-prefill, was built as
  the strip. Gemma stays thinking-off.
- The rest of this page is the analysis of 2026-10-30, read at `cf2b28d`.

## Status (2026-10-30): stage 1 implemented on branch thinking-on, CPU-tested only

`serve.py --enable-thinking` (opt-in; Qwen, `mswea-xml` and `chat-text`; [docs/running.md section 6.8](../running.md#68-thinking-on-stage-1---enable-thinking-not-validated),
`remote_ra_live_v2/thinking.py`, tests `test_thinking_on.py`): the thinking-on header and template arguments
pinned in `startup.json` / `ready.json`; the response split (content C, `reasoning_content` and `reasoning` R;
no `</think>`: all reasoning, content `""`); MarkerStop armed only after `</think>`; an append when the client
sends R back (content and reasoning compared after `strip()`), the existing rebuild otherwise (stage 2, cut and
re-prefill, not implemented); the row's `thinking` record; the Goal's previous reply C only (flagged for the
Goal owner). Not GPU-validated; no benchmark run. Gemma stays off; kv-session unchanged.

## Addendum (2026-10-30, probe results; supersedes the comparability statements below)

Probe: its scripts, result logs and check notes are a local scratch copy, not in this repository (ask the
maintainer). Read-only WebFetch of the collaborator's repo token_efficiency @ 068efec, and a stub server on the agent
host.

- **The collaborator's baselines run thinking ON** (from that WebFetch read; their repository is not part of this
  one, and this was not re-checked against a local copy) (configs: enable_thinking true; max_tokens 4096 incl.
  thinking). They render prompts themselves (tebench/backends/base.py: apply_chat_template with enable_thinking; no
  preserve_thinking anywhere; generated text split at `</think>`). AgentKV, RepairKV, Sliding Window and Random use a
  kv-session backend (raw tokens appended, reasoning kept). Vanilla and IntentKV re-render the whole history each
  turn:
  - Terminal-Bench (Terminus 2, interleaved_thinking True, observations role user): the stock template drops every
    earlier reasoning, the last reply's included -> **R dropped** (the stock default).
  - SWE-bench (mini-swe-agent 2.4.6 tool-calling, reasoning_content sent back, observations role tool): the template
    keeps reasoning after the last real user message -> **R kept** (unless a format-error user message arrives).
  - MemoryAgentBench: R kept while reading (chunks role tool), dropped at the question (role user).
  So "default baselines drop R" (sections 3-4 below) holds only for their Terminal-Bench runs.
- **Our clients against a stub** (A: `R\n</think>\n\nC` inline in content, what the runtime returned at `cf2b28d`
  (mswea_route.py ~49-52, chat_text_route.py ~176-181 blanked the reasoning fields; since stage 1, `--enable-thinking`
  returns C in content and R in `reasoning_content` and `reasoning`); B: reasoning_content; B2: reasoning):
  - Mini-SWE-Agent 2.4.2 (through litellm; its version is not in the probe's logs): A keeps R inline in content; a
    full `<mswea_bash_command>` draft inside R gives a FormatError and the next request drops that reply (not an
    append). B / B2: R sent back as a separate reasoning_content field (litellm maps `reasoning` to
    reasoning_content); actions parse.
  - Terminus 2 (Harbor 0.21.0, litellm 1.97.0): A breaks parsing (the parser took a JSON object from R; turn 2 executed
    a draft command from R). B / B2 with interleaved_thinking off (our default): only {role, content} sent back, R
    dropped; on: reasoning_content sent back as a separate field. Actions parse.
  - Neither client sends chat_template_kwargs; neither sends a top-level `reasoning` key, which is the only field our
    overlay's chat_utils reads (stock vLLM 0.24 would ignore the returned reasoning_content; not checked upstream).
- **Consequences:** every option needs the server to split reasoning from content in its response (A breaks Terminus
  and gives Mini format errors), and MarkerStop must ignore markers inside R. Which history semantics give Ours the
  same input as the collaborator's Vanilla depends on the benchmark: Terminal-Bench drops R (needs option (a) cut and
  re-prefill), SWE-bench with their tool-calling harness keeps R (append-only works once the reasoning is split and
  sent back), MemoryAgentBench keeps R until the question. Note their SWE harness (mini 2.4.6 tool-calling) differs
  from ours (2.4.2 XML PoC prompt).

## Verdict

The group's intuition is half right: with the stock template the next prompt diverges from the previous turn only at
the last reply, so earlier history's KV is unaffected. But this runtime cannot use that today: any request that is not
an append rebuilds the whole session; there is no "cut only the last reply" path.

## 1. The next prompt after a thinking-on turn

- Thinking-on header `<|im_start|>assistant\n<think>\n` (5 tokens; off: 7). The model generates `R\n</think>\n\nC<|im_end|>`
  (R = reasoning, C = visible reply). `<think>` (248068) and `</think>` (248069) are not special tokens, so after
  skip_special_tokens the content still holds `R\n</think>\n\nC`.
- History rendering: without preserve_thinking, an assistant turn before the last real user message renders as
  `<|im_start|>assistant\nC<|im_end|>`; even if the client sends `R</think>C` back, the template cuts R at `</think>`.
  With preserve_thinking (R in reasoning_content or inline) the rendering equals the server's append stream token for
  token (checked by the report's local script, which was not kept; not re-checked).
- Divergence: the common prefix with the server stream ends at the last reply's `<|im_start|>assistant\n` (header start
  + 3); removed: `<think>`, `\n`, R, `\n</think>\n\n` (~|R| + 5 tokens); C is re-encoded from text (usually the same ids,
  not guaranteed). Earlier replies have no R on either side (not re-checked).

| Protocol / harness | What the client sends back | Against the server stream |
| --- | --- | --- |
| mswea-xml + Mini-SWE-Agent | depends on the server's response | below |
| chat-text + Terminus 2 | only C by default | the divergence above |
| kv-session (MemoryAgentBench) | only the tail | none |

- mswea-xml: matches exact generated ids by content.strip(), not by template rendering. If the server returns
  `R</think>C` in content (as at `cf2b28d`), litellm probably passes it through (it splits reasoning only when content
  starts with `<think>`, which is in the prompt) -> still append-only but R stays in history; MarkerStop would stop
  early on a `</mswea_bash_command>` inside R, and the agent's action regex would scan R. If the server splits
  reasoning_content, the agent sends back only C -> the divergence. A stock control drops R through the template in
  both cases.
- chat-text / Terminus 2: content must be C only (the JSON parser takes the first balanced `{...}`; braces in R break
  it). Default interleaved_thinking=False -> only C -> divergence. Harbor 0.21.0 has an interleaved_thinking kwarg that
  puts reasoning_content in history (harbor/llms/chat.py); whether litellm's hosted_vllm forwards it is unverified.
- kv-session: the client sends only the tail; the server stream keeps R, i.e. the preserve_thinking rendering
  (test_kv_session_protocol.py). Only the Vanilla control differs: at the question turn it drops all earlier R.

## 2. What the runtime did at cf2b28d, and the cost

- At `cf2b28d`, mswea-xml and chat-text could not turn thinking on: header and template kwargs were fixed; the
  client's chat_template_kwargs had no effect. Since stage 1, `--enable-thinking` pins the thinking-on header and
  template arguments for both protocols on Qwen (`remote_ra_live_v2/thinking.py`).
- Changing only the header while the client strips R: extends() fails every turn -> rebuild every turn (close_session
  frees resident Pages, tail and GDN carry; selector and Goal reset; full re-prefill from 0; local turn 0 -> cold_start
  selection, all pages active: the query policies degrade to full every turn). No partial-truncation path
  (turn_geometry requires T[:C] == computed, else SESSION_PREFIX_MISMATCH; [docs/running.md section 6.3](../running.md#63-errors) lists history mutation as a hard
  failure).
- If the last reply were cut and re-prefilled: re-prefill |C| + 3 tokens per turn (thinking-off mean replies: SWE ~223
  tokens, job 3404230: 500,619 / 2,245; TB ~557, job 3434533: 195,484 / 351) - negligible against prompts of tens of
  thousands of tokens. Wasted: R's KV and ~|R|/16 sealed Pages (1 MiB GPU each + an arena host copy + vectors; ~|R|/16
  MiB of D2H per turn). GDN needs the state at the cut point e (the phase-1-end probe state, not committed today): a
  new rollback carry, ~154 MB pinned per session. The tail Page across the cut was sealed during decode with obs,
  header and R in its vector: unseal back to the tail (valid_end = e) and drop the vector. Selection geometry returns to
  the history without R (as thinking-off); the recent set = obs Pages + the re-prefilled C Pages (attributed to the
  previous turn); mandatory unchanged.
- If R stays in history: no recompute or rollback, but +|R| + 5 tokens per turn: Pages, candidates, arena and context
  grow faster (Terminus summarizes earlier -> more rebuilds; lower active fraction at a fixed budget; the 1024-token
  recent set mostly the previous reasoning).
- All options: the reply bound has only 1536 / 4096 (thinking needs a larger choice: MAX_TOKENS_CHOICES, the capacity
  rule ceil(max_tokens / P), pinned host budget, context checks); the Goal's previous reply should use C only.

## 3. Options (by real invasiveness in this runtime; append-only is an engine invariant, so (a) is the most invasive)

- (c) Server keeps R: the last assistant content equal to the stored C counts as an extension, the stream keeps R's
  exact ids. Protocol layer only: header on; response split into content + reasoning_content; MarkerStop only after
  `</think>`; new extends rule; new reply-bound choice; evidence records R; the offline reference renders with the
  preserve_thinking template. Engine unchanged. Cost: the model's input differs from the client's messages.
- (b) Client sends R back + preserve_thinking: (c) plus Terminus interleaved_thinking: true; TEMPLATE_KWARGS
  {enable_thinking: true, preserve_thinking: true}; extends compares (reasoning_content, content); a rebuild's template
  rendering equals the stream. A stock baseline with the same settings is token-comparable. Mini-SWE-Agent has no such
  switch (R only inline in content).
- (a) Cut the last reply and re-prefill (strip semantics): a rollback-to-e geometry path, GDN rollback carry, unseal,
  Page release (GPU, arena, idle release, allocation owner), selection and turn-record rollback; evidence
  truncate{cut, dropped, reprefill}; the verifier changes from "each token counted once" to "each stream version
  counted once". Matches stock default thinking-on token for token (re-encoding C from text would even remove the
  4-token-per-reply difference seen with thinking off).
- Tests: CPU - pinned-tokenizer template equivalence (enable_thinking=True, preserve on/off), the TB ATIF 178-trial
  replay, a Mini-SWE-Agent replay; (a) also engine fakes across the fast paths, admission overlap, deferred IO, idle
  release. GPU - one SWE and one TB dry run per policy; (a) also token alignment against a stock eager replay.
- Gemma: its template always strips thinking for text-only turns (preserve_thinking only for tool_calls), thinking on
  injects `<|think|>` into the first system turn (changes the mandatory prefix), and the window layers' rollback differs
  from GDN: (b) impossible, (c) has no template reference, (a) separate work. Recommendation: Gemma stays off. (Superseded on 2026-11-09: the maintainer chose (c) and (a) for Gemma, [docs/thinking-gemma/DESIGN.md](../thinking-gemma/DESIGN.md).)
- Comparability: none of the options is comparable with past thinking-off runs. Against a collaborator's thinking-on
  baseline with stock + default template / default harness (history drops R) only (a) gives the same input; (b) and (c)
  compare only with preserve_thinking baselines.

## 4. Effort (rough) and recommendation

| Option | Effort | Risk |
| --- | --- | --- |
| kv-session | no code, 1 GPU job (the Vanilla control needs preserve_thinking to be comparable) | low |
| (c) | 1-2 days CPU + dry runs | low |
| (b) | (c) + 0.5-1 day | low |
| (a) | 1.5-3 weeks + several GPU jobs | highest |

First, two checks without a GPU: (1) the collaborator's thinking-on baseline rendering and history settings; (2) on
the agent host, a stub server to see what Mini-SWE-Agent / litellm and Harbor with interleaved_thinking actually send back.
Then: token-exact comparability with default thinking-on baselines -> (a); only showing KV reuse works with thinking on,
or baselines can switch to preserve -> (b) for TB, (c) for SWE, MAB straight to a GPU check.
