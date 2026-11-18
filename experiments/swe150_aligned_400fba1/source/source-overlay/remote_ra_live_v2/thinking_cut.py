"""--thinking-history strip: thinking on, option (a) form E, "commit at the cut" (chat-text, Qwen).

Design and decisions: docs/thinking-cut/DESIGN.md. Functional GPU checks passed (jobs 3493649,
3500646 and 3499759); no full benchmark result yet. Without the option (serve.py
--thinking-history absent) none of this runs: stage 1's keep mode (remote_ra_live_v2.thinking)
and every startup config, evidence byte and response stay as they were.

The reference is the stock default template with ``enable_thinking=True`` and no
``preserve_thinking`` (the collaborator's Vanilla on Terminal-Bench): every assistant message
before the last user message renders as ``<|im_start|>assistant\\n`` + its content only, the
reasoning dropped, whatever the client sends back. So on a server pinned to this history every
continuing chat-text request diverges from the session's stream at X = e_t, the end of the last
observation span (the start of the last reply's generation header, N_t - |header|). The
completion commits the session there; phase 2 (header, R, C, EOS) is turn-local scratch:

- The transcript (chat_text_protocol.ChatTextTranscript, ``strip``): after a reply the stream is
  prompt[:X]; the turn record keeps the dropped branch (header + generated ids) and the cut. A
  continuing request renders ``stream + A + U + header``: A = encode(``reply_piece(c)``) of the
  content the client sent back (``strip_content`` is the template's own transform: trim, then
  the text after the last ``</think>`` with its leading newlines removed), U the user piece. Its
  stable end is X + K, K = len(encode(REPLY_HEAD)) (3 on the pinned Qwen3.6 tokenizer, the
  header's first three tokens), and its observation span (X + K, X + |A| + |U|): the turn geometry
  is unchanged and CANDIDATE_RANGE is not relaxed. Each continuing turn compares A's first K
  tokens with the header's (K tokens, no hashing) and refuses the request when they differ.
- The engine (remote_ra_native_growing.adapter.EngineCoreAdapter, ``commit_at_cut``): the plan
  names ``commit_end`` X and the seal rule floor(X/P); the completion seals [L/P, floor(X/P)) plus a
  phase-1 tail candidate, pins resident = selected | [L/P, floor(X/P)) and the tail page X//P with
  valid_end X (none when X % P == 0), sets computed = T[:X], sealed = T[:floor(X/P)P],
  next_token = T[X], and checks the carry's computed_end X; no phase-2 Page is sealed, captured or
  published. The worker's receipt keeps computed_end C' = N + g - 1 (what phase 2 computed).
- The worker (allocation_owner, worker_backend, qwen; ``commit_at_cut``): the same seal, tail and
  resident set at X; the committed GDN carry is a positioned carry at X (worker_backend.
  PositionedCarry): at a local turn > 0 the phase-1 state at e (its payload and identity token,
  no copy), at local turn 0 the state captured at the end of the prefill forward that ends at X.
  The C' capture is skipped and phase 2 compacts no Page vector at or past the cut's Page.
- Local turn 0: EngineBridge.scheduler_split ends a prefill forward at X (``split``), after
  vLLM's Mamba-aligned split (scheduler.py _mamba_block_aligned_split): a chunk that holds X
  strictly inside ends there. Before the last M multiple (N mod M < |header|) the next chunk
  [X, floor(N/M)M) then ends at that multiple by vLLM's own rule, and [floor(N/M)M, N) follows.
  The chunk that ends at X stays in the align block of X - 1 (preprocess_mamba's slot
  cdiv(X, M) - 1, the same block as the chunk after it), as any chunk of the last partial block
  does today; the exactness of that state on the GPU is check G1.
- Evidence (remote_ra_http.evidence): the native file adds ``truncate`` {cut, dropped {start,
  end, header_ids, generated_ids, think_end}} and, for a continuing turn, ``reprefill`` {start,
  piece_end}; its plan names commit_end and floor(X/P); the runtime row's stage_b computed_end and
  sealed_end are the committed ends X and floor(X/P)P, and it adds phase2_end (C') and ``truncate``
  counts; the row's ``thinking`` adds cut and reprefill.

The positioned carry (worker_backend: ``hold_carry``, ``drain(..., positioned=True)``) commits a
GDN state at a named end X <= the request's cursor with a named source ('phase1', 'split');
async scheduling option C can add a third source (a snapshot before a speculative step) and
commit at X < cursor the same way (docs/thinking-cut/DESIGN.md section 2.3).

Pinned per server at startup (``pinned``, ``loaded``, ``worker_flag``); refused on kv-session and
mswea-xml, without --enable-thinking, and in v1 with --gdn-device-handoff and --phase1-in-step. A
re-prefill [X, e) longer than one native forward fails the turn as any over-long observation does
(PHASE1_TOO_LONG). Nothing is hashed per request beyond what a turn without the option hashes.

The Gemma 4 profile (gemma-thinking; docs/thinking-gemma/DESIGN.md sections 3.3 and 4): the same
transcript with Gemma's pieces. A past reply renders as ``<|turn>model\\n`` + the template's
``strip_thinking`` of its content + ``<turn|>\\n`` (``gemma4_reply_piece``), whose head is the whole
thinking-on header (K = |H| = 3 on the pinned tokenizer), so the stable end is X + 3 as on Qwen. In
v1 it refuses --async-scheduling (``check_config``). What crosses turns on its window layout is the
sliding window, which phase 2 frees while it decodes past X, so the engine and the worker hold the
window at X (option G, remote_ra_live_v2.window_hold): ``attach_engine`` and ``attach_worker`` hand
a window-layout server to window_hold's attachments, and ``check_config`` requires the layout's
hold slots (``kv_layout.hold_slots`` H >= 1, serve.py --window-hold-slots; at most H strip turns in
flight, each from its admission to its commit). CPU-tested only, not on a GPU.
"""
from __future__ import annotations

import os

from .contract import require

STRIP = 'strip'                         # serve.py --thinking-history strip
TEMPLATE_KWARGS = {'enable_thinking': True}     # the stock default (no preserve_thinking)
GEMMA4_TEMPLATE_KWARGS = {'enable_thinking': True}      # the Gemma 4 profile's stock thinking-on arguments
RULE = 'previous_messages_exact+reply_content_stripped;cut_at_header_start+reprefill_reply_sent_back'
WORKER_KEY = 'thinking_history'         # worker startup record and its echo ('strip'; off: absent)
REPLY_HEAD = '<|im_start|>assistant\n'  # a past reply's turn head: the header's first tokens
REPLY_END = '<|im_end|>\n'
THINK_END = '</think>'
# The Gemma 4 profile (docs/thinking-gemma/DESIGN.md): a past reply's turn head is the whole thinking-on
# header (K = |H|, 3 tokens on the pinned tokenizer), its end the template's end of turn.
GEMMA4_REPLY_HEAD = '<|turn>model\n'
GEMMA4_REPLY_END = '<turn|>\n'
# How a request's prompt was built (the row's thinking.history on a strip server).
FRESH, CUT, REBUILD = 'fresh', 'cut', 'rebuild'


# ---- the template's history render ------------------------------------------------------------

def strip_content(content):
    """The pinned template's render of an assistant message's content when the message has no
    reasoning field (thinking off for the history): ``|trim``, then, when ``</think>`` is in it,
    the text after the last ``</think>`` with its leading newlines removed (Python's str
    methods, as the template calls them)."""
    require(type(content) is str, 'assistant content must be a string')
    text = content.strip()
    if THINK_END in text:
        text = text.split(THINK_END)[-1].lstrip('\n')
    return text


def reply_piece(content):
    """A past reply as the template renders it before the last user message: the turn head,
    the stripped content and ``<|im_end|>\\n`` (each piece starts and ends at a special token,
    so it encodes alone exactly as inside a whole render)."""
    return REPLY_HEAD + strip_content(content) + REPLY_END


def gemma4_strip_content(content):
    """The pinned Gemma 4 template's render of an assistant message's content in the history: its
    ``strip_thinking`` (every ``<|channel>...<channel|>`` part removed, then trimmed), whatever reasoning
    the message carries (the template drops the reasoning of every reply before the last user message)."""
    require(type(content) is str, 'assistant content must be a string')
    from .mswea_protocol import strip_thinking
    return strip_thinking(content)


def gemma4_reply_piece(content):
    """A past reply as the Gemma template renders it with thinking on: ``<|turn>model\\n``, the stripped
    content and ``<turn|>\\n`` (no thought channel; each end is a special token, so the piece encodes alone
    exactly as inside a whole render)."""
    return GEMMA4_REPLY_HEAD + gemma4_strip_content(content) + GEMMA4_REPLY_END


def history_of(transcript, extends):
    """How a request's prompt is built on a strip transcript (evidence): fresh, cut (the
    previous reply re-prefilled at the cut) or rebuild."""
    if transcript.turn == 0:
        return FRESH
    return CUT if extends else REBUILD


# ---- the pin (startup only) ---------------------------------------------------------------------

def check_config(config):
    """True when a startup config pins --thinking-history strip, else None. Refuses it on any
    protocol but chat-text, with --gdn-device-handoff or --phase1-in-step (v1), and on the Gemma 4
    profile with --async-scheduling (v1: async C voids rows in every sliding-window group; decision 7
    of docs/thinking-gemma/DESIGN.md) or without hold slots in its window layout (window_hold: the
    sliding window at the cut is held in one of H slots; H = 0 could not decode a reply). Startup only."""
    from .thinking import HISTORY_KEY, CONFIG_KEY as THINKING
    value = None if config is None else config.get(THINKING)
    if type(value) is not dict or HISTORY_KEY not in value:
        return None
    from .thinking import pinned as thinking_pinned
    thinking_pinned(config)          # the record itself (the profile's, chat-text, history strip)
    require(config.get('protocol') == 'chat_text_v1', '--thinking-history strip serves the chat-text protocol only')
    if config.get('model_profile') is not None:
        require(config.get('async_scheduling') is not True,
                '--thinking-history strip on the Gemma 4 profile refuses --async-scheduling (v1)')
        from .kv_layout import HOLD_FIELD
        layout = config.get('kv_layout')
        hold = layout.get(HOLD_FIELD) if type(layout) is dict else None
        require(type(hold) is int and hold >= 1,
                '--thinking-history strip on the Gemma 4 profile needs hold slots for the sliding window at '
                'the cut (kv_layout.%s >= 1, serve.py --window-hold-slots): with 0 hold slots no strip reply '
                'could decode' % HOLD_FIELD)
    from .admission_flags import on, pinned as admission_pinned
    flags = admission_pinned(config)
    require(not on(flags, 'gdn_device_handoff') and not on(flags, 'phase1_in_step'),
            '--thinking-history strip refuses --gdn-device-handoff and --phase1-in-step (v1)')
    return True


def pinned(config):
    """True when the startup config pins the strip history, else None."""
    return check_config(config)


def loaded():
    """The pin of this process's startup config on the HTTP runtime, else None."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def worker_flag(startup):
    """Worker startup: the record the engine sent ('strip' or absent), checked once against the
    pinned config. Returns True or None."""
    sent = startup.get(WORKER_KEY)
    require(sent is None or sent == STRIP, 'thinking_history worker record must be strip or absent')
    on = True if sent == STRIP else None
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == on, 'worker thinking_history differs from the pinned startup config')
    return on


# ---- the attachments ----------------------------------------------------------------------------

def attach_engine(adapter, bridge):
    """compose_engine: the completion commits at the cut, and a local turn 0 prefill ends a
    forward at X (EngineBridge.scheduler_split). The window layout (the Gemma 4 profile): the
    completion commits at the cut with the sliding window held at X (window_hold.attach_engine, option
    G; no split: the window's KV is exact whatever chunk computed it)."""
    require(not getattr(adapter, 'commit_at_cut', False) and not getattr(bridge, 'commit_at_cut', False),
            'thinking_history engine attachment collision')
    if getattr(adapter, 'window', None) is not None:
        from .window_hold import attach_engine as attach_window
        return attach_window(adapter, bridge)
    adapter.commit_at_cut = True
    adapter.cut_records = {}            # request -> its committed cut (the native file's truncate)
    bridge.commit_at_cut = True


def attach_worker(owner, qwen, backend):
    """attach_loaded_worker: the worker owner, the Qwen lifecycle and the backend commit at X. The
    window worker (attach_window_worker, the Gemma 4 profile): its owner, lifecycle and backend commit
    the window held at X (window_hold.attach_worker)."""
    require(not getattr(owner, 'handoff', False) and not getattr(backend, 'handoff', False),
            '--thinking-history strip refuses --gdn-device-handoff (v1)')
    require(not getattr(owner, 'phase1_in_step', False), '--thinking-history strip refuses --phase1-in-step (v1)')
    if getattr(backend, 'window', False) is True:
        from .window_hold import attach_worker as attach_window
        return attach_window(owner, qwen, backend)
    owner.commit_at_cut = qwen.commit_at_cut = backend.commit_at_cut = True


def split(assignment, start, count):
    """EngineBridge.scheduler_split on a strip server: a local turn 0 prefill chunk that holds
    X = e strictly inside ends at X; anything else is the chunk vLLM proposed."""
    if assignment.local_turn != 0:
        return count
    cut = assignment.observation_end
    if start < cut < start + count:
        return cut - start
    return count


# ---- evidence -------------------------------------------------------------------------------------

def think_end(generated, ids):
    """The index of the first ``</think>`` id in the generated ids when ``</think>`` is one
    token (``ids``, the transcript's encoding, startup), else None; None without it."""
    if ids is None or len(ids) != 1:
        return None
    try:
        return generated.index(ids[0])
    except ValueError:
        return None


def native_fields(core, binding):
    """The native file's ``truncate`` and ``reprefill`` of a completed request on a strip
    server ({} otherwise): the cut the engine committed (EngineCoreAdapter.cut_records) and
    the re-prefilled piece of the engine transcript's committed turn."""
    adapter = core._remote_ra_growing_adapter
    if not getattr(adapter, 'commit_at_cut', False):
        return {}
    record = adapter.cut_records.get(binding.request)
    require(record is not None, 'strip completion without its cut record')
    transcript = core._remote_ra_private_v2_endpoint.transcripts[binding.session]
    turn = transcript.turns[-1] if transcript.turns else None
    require(turn is not None and turn.get('cut') == record['cut'] and
            list(turn['ids'][len(record['header_ids']):]) == record['generated_ids'],
            'engine transcript turn differs from the committed cut')
    dropped = dict(start=record['cut'], end=record['end'], header_ids=list(record['header_ids']),
                   generated_ids=list(record['generated_ids']),
                   think_end=think_end(record['generated_ids'], transcript.think_end_ids))
    value = dict(truncate=dict(cut=record['cut'], dropped=dropped))
    if 'window_hold' in record:
        # The window layout (window_hold): where the window at the cut came from.
        value['truncate']['window_hold'] = dict(record['window_hold'])
    if turn.get('reprefill') is not None:
        value['reprefill'] = dict(turn['reprefill'])
    return value


def row_fields(native):
    """The runtime row's extra fields from a strip native file: (stage_b additions, the row's
    ``truncate``); ({}, None) for any other native file."""
    truncate = native.get('truncate')
    if truncate is None:
        return {}, None
    done = native['worker_frames'][-1]
    dropped = truncate['dropped']
    row = dict(cut=truncate['cut'], dropped_tokens=dropped['end'] - dropped['start'],
               header_tokens=len(dropped['header_ids']), generated_tokens=len(dropped['generated_ids']),
               think_end=dropped['think_end'], reprefill=native.get('reprefill'))
    if 'window_hold' in truncate:
        row['window_hold'] = dict(truncate['window_hold'])      # the window layout (window_hold)
    return dict(phase2_end=done['computed_end']), row
