"""perf-swe16 library copy: perf-adm's and perf-ctx's smoke_checks merged over perf-int (git merge-file; both
feature sets, as int-1010 merged the runtime), plus two perf-swe16 edits: check_gdn_bytes and check_phase1_mode key
the GDN carry and phase 1 on stage_b.local_turn (a real agent's session may rebuild). swe_checks.py composes the
evidence-only checks below for real-agent session-pool servers; the client-driven checks (http, token-stream of the
synthetic driver, join, queue waits, check_policy, check_plan) are kept unused.

Per-policy checks of one multi-session stage-2 server run: READY and the batch configuration, routing,
per-session HTTP, token stream, Stage B turn geometry, selection, the request's own IO and admission stall,
disjoint evidence, serialized admissions, capacity waits, no preemption, cleanup.

The per-request checks are those of the v8 dev smoke (job 3265689), the v9 clean acceptance (job 3270721),
the v10 measurements and measure-ms1, run per session: each session's requests are joined to the server's
evidence of that session and must form their own token-exact transcript, Stage B chain and selection
history. The stage-2 evidence follows development/multi-session/STAGE2_CONTRACT.md in the format of the
runtime branch ms-stage2 (4aa29bc: ms2-admission 35bcf77 + ms2-worker 53a9dd2):
- READY's effective server config runs one sequence per session (`max_num_seqs` = N) on the pinned KV budget;
- session ids derive from the server id (READY `session_id`): session 0 keeps it, session k is
  `<server id>-s<k>` (`remote_ra_live_v2.sessions.session_ids`); a request is routed by its task message;
- every runtime row carries `session_id`, `session_index`, the per-session `turn` (`ordinal == turn + 1`),
  `request_index` (the server-wide order of first admission), `queue_wait_s` (the admission-lock wait plus
  every CAPACITY_WAIT retry) and `capacity_waits`; rows are written at completion, so under batched
  decode they are not in admission order;
- native terminal records carry the session id, its index, the per-session turn and `admission_engine_ms`
  (that admission's wall time on the engine thread: the decode stall); their worker frames carry the
  request's own cumulative H2D counters, zero before its first frame (remote_ra_growing_worker.tensors
  `attributed`), and the complete frame the KV blocks of that request's generation only;
- one server-wide FIFO admission lock, released at engine acceptance: admissions never overlap in the
  server log; a refused admission is the capacity gate's authenticated CAPACITY_WAIT (engine_rejected
  `CapacityWait: CAPACITY_WAIT ...`), retried from capture and counted in the row's `capacity_waits`;
- no preemption: vLLM's preemption counter stays 0 in every /metrics scrape of the server;
- each session keeps one KV namespace, the one engine-cleanup.json closes for it.
The Goal rewriter follows the pinned runtime (job expectation `rewriter_policies`): without commit 84ed862
both query policies use it; with it only goal_query does, observation_query's contract pins none, its
turns never attempt it and a REWRITE gate falls back with `rewriter_disabled`.

ms2a (runtime 9b18421, remote_ra_live_v2/goal_async.py): on a server with N > 1 sessions and a Goal rewriter
(job expectation `parking_policies`) every goal_query admission parks at its Goal; the Goal is built on a
background thread while the engine thread keeps decoding, and the admission resumes on the engine thread
before its ack. An admission is then the union of its engine-thread pieces (dispatch up to the park, resume
up to the ack): `admission-log` requires the engine-thread pieces never to overlap (a parked interval may
contain decode steps and other requests' completions); `goal-async` requires each parked admission to have
exactly one resume and one ack, in order, every accepted admission of a parking policy to park and no other
policy to park, and the native records' `goal_async_ms` / `goal_construct_ms` to match; `goal-fallback`
reports a Goal that fell back (a rewriter timeout or failure: `GOAL_FALLBACK`) as a finding, never a failure.

smoke-dp1 (decoupled FA Pages, docs/exp-decouple-pages/DESIGN.md): every check runs at the server's own Page
size P (job expectation `layout`: the legacy 784-token shared pool, or a decoupled layout with P-token FA Pages
and a separate Mamba pool): turn geometry, candidate Pages, the Selector V2 contract (P = 16: recent quota 64,
budgets floored to 16 tokens, vector format `compact_int8_v1`) and the request's active KV (FA blocks of
P x 65,536 B, Mamba-pool blocks of 51,314,688 B). New per-server checks: `kv-layout` (READY / startup kv_layout,
both pools' sizes per DESIGN D4 and the readiness baseline; a legacy server has none of these fields),
`kv-pools` (every row's pool sample within its pool, engine-cleanup.json back at the readiness baseline after
every session closed), `startup-log` (legacy logs vLLM's 784-token alignment and Mamba padding, a decoupled
server neither). check_plan runs over all runs of a plan: token identity against the reference run, the
selection numbers at P = 16, restores, and the per-step host overhead against the reference.

Pure functions over the files of one server output directory (client files under s<k>/) and its
server log. The GPU job runs them after each server has stopped; postrun/verify_evidence.py runs them
again offline.

perf-ms16 (one segment of runs with their own session counts, 4 and 16): check_plan adds `batched` at 16 sessions
(the 1-s metrics samples reach 16 running requests on every 16-session run, with vLLM's own "Running: 16 reqs" log
lines counted; otherwise the finding NOT_BATCHED_16, a partial pass), `cross-run` (sessions 0-3 of S16-full against
R4: identical turns, or the finding BATCH_SHAPE_DIVERGENCE at the first reply that differs; a prompt that differs after
identical replies is a failure) and `residual` (the admission residual admission_engine_ms - phase1_ms against the
candidate Pages, least squares per run: the finding RESIDUAL_SLOPE_HIGH above the pinned microseconds per Page). The
stock-vLLM checks of perf-p16 are gone.

``tokens`` is an exact tokenizer (HFTokens: encode, decode, chat template) or the
decode-only VocabDecoder. Decode checks on the server's own boundaries (stable_end,
observation_span) run in both modes; token-exact encode checks need an exact tokenizer.

perf-adm (adm-e df16cd9: (e) `--gdn-device-handoff`, (b) `--phase1-in-step`; job expectation `admission`, the run's
record {flag: True}, None when every flag is off). With every flag off every check is perf-int's, plus `gdn-bytes`'s
flag-off identities. Per server:
- `admission-flags`: startup.json, READY and the worker's attach receipt carry exactly the run's `admission` record
  (none of them when off);
- `gdn-bytes` (decoupled): per turn, prepare H2D = the GDN carry (0 at turn 0); off: restore H2D = the restored Pages
  + GDN x (1 + crossings) (turn 0: GDN x crossings) and no D2D field; with the hand-off: restore H2D = the restored
  Pages only, restore D2D = GDN x (1 + crossings) (turn 0: GDN x crossings), every frame's counters carry d2d_bytes,
  zero before the request's first frame, never decreasing, and the row's D2D is the frames' own. Crossings are the
  multiples of the Mamba block M strictly inside phase 2 [phase2_start, computed_end) (one more allowed when phase 2
  starts on a multiple): the off identity holds on all 488 GPU turns of jobs 3339019 and 3339448;
- `phase1-mode`: off: no phase1_mode / phase1_park_ms / phase1_step_ns / phase1_step_rows and no phase1_arm frame;
  (b) on a one-session server (no park table: every phase 1 is the separate probe-only pass, bit for bit): the same
  frames as off and phase1_mode absent or probe_only on every turn, nothing parked; (b) with several sessions: every
  continuing turn names its mode (mixed, probe_only, fallback); a mixed turn parked, ran in a step of 2..S rows
  (phase1_step_rows) for phase1_step_ns, and its frames are prepare, phase1_arm, phase1 (measured from the arm:
  its `before` is the arm's `after`, so prepare H2D stays the carry), phase1_drain, reserve; a turn that did not run
  mixed has no arm frame and no step fields;
- the frame order and the IO counters accept the arm frame and the D2D counter exactly where the flags allow them;
  `goal-async` accepts phase-1 parks on an in-step server with several sessions (an admission parks iff its native
  record carries phase1_park_ms); `kv-pools` allows three GDN slots per session with the hand-off (the held probe slot).
check_plan adds `parity` (S = 1: tokens, selections and int8 Page vectors of each flag run equal off's: a failure),
`repeat` (off again against off: a difference is the finding EAGER_REPEAT_DIFFERS; per-Page cosine quantiles either
way), `flags-compare` (S > 1: tokens and selection agreement against off as findings FLAG_*), `capacity-waits` (a flag
run's CAPACITY_WAITs at most off's) and `adm-report` (stall share, mixed turns, probe-step time against plain decode
steps, ITL and tok/s at S, bytes per turn).

(f) (adm-f 4eae441, `--admission-overlap K`; the record gains `admission_overlap` = {permits: K, admission_seconds:
300 + 120 (K - 1), lease_seconds: 360 + 120 (K - 1)}): `admission-flags` checks it like the switches; `admission-log`
lets up to K admission blocks overlap (K read from the blocks' pinned ACK wait, `measure.log_permits`) where off keeps
them disjoint; `goal-async` requires the log's K to be the run's, accepts several admissions parked at once (a resume
that acknowledges nothing while several are parked is reported as unattributed engine time, an ADD the log cannot
name as unattributed, neither a failure), and `lease_wait_ms` only on a server with (f) and (b) (finite, >= 0);
`goal-order` (goal_query): each session's native records name its own session and turns 0, 1, ... in order with a
Goal record on each. Plan checks: `flags-compare` adds the Goal decisions and Goal hashes of a goal_query flag run
against off where the prompts coincide (FLAG_GOAL_DIFFERS, a finding); `capacity-waits` of a run with (f) is the
finding OVERLAP_CAPACITY_WAITS (each admission in flight reserves its blocks, with (e) three GDN slots of 2S + 1), not
a failure; `adm-report` adds the (f) columns (measure.overlap_summary).
"""
import hashlib
import json
import math
from pathlib import Path
import re
import traceback

import measure

PAGE = 784  # the legacy Page; a decoupled server's P comes from the job expectation `layout`
SWITCH = 8192
PROTOCOL = 'mswea_xml_v1'
HEADER = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
IM_END_NL = '<|im_end|>\n'
MARKER = '</mswea_bash_command>'
MAX_TOKENS = 1536
MAX_BATCHED = 16384
QUERY = ('goal_query', 'observation_query')
# perf-p16: the random-fill control of the sealed runtime (exp-random-fill 780fe54; pilot-v14 pilot_checks): checked
# like observation_query (contract, issue and observation linkage, Goal records without the rewriter) except its fill:
# the contract carries `random_fill` {seed, rng}; an active turn draws exactly the first capacity - |mandatory| -
# |recent| Pages of fill_order(candidates - mandatory - recent, seed, session index, turn), with no projection, query
# hash, score or Goal forward.
CONTROL = ('random_fill',)
SELECTIVE = QUERY + CONTROL
RANDOM_FILL_RNG = 'splitmix64_fisher_yates_v1'
DEFAULT_RANDOM_FILL = dict(seed=42, rng=RANDOM_FILL_RNG)
GOAL_REWRITER_UNUSED_RANDOM_FILL = 'random_fill fills the remaining Pages at random and ranks nothing'
_MASK64 = (1 << 64) - 1
_GOLDEN64 = 0x9E3779B97F4A7C15
# Session ends that are not failures: planned turns done, or truncated by the time or context guard.
STOPS_OK = ('completed', 'skipped_time_budget', 'skipped_context_budget')
PROJECTION_SHA256 = '20c2e1cf4463ed9407a9f9af0a7794fe044c02c0da369128681270a7427c87aa'
# serve.py selection_contract(): contract.default_contract(784, active_ratio=.4, goal_rewriter=rewriter_pin(path))
CONTRACT = dict(schema='remote-ra-selection-contract-v1',
                upstream='dram-backed-kv scripts/serve_online_selector_v2.py a70701e', page_tokens=PAGE,
                switch_threshold_tokens=SWITCH, active_ratio=0.4, active_token_budget=None, recent_turns=2,
                recent_quota_pages=2, mandatory='initial_prompt_pages', projection_dimension=128,
                projection_seed=20260904, vector_dtype='int8', hidden_state='pre_final_norm',
                goal_query_variant='full_goal', entity_drift_steps=3)
REWRITER_DEFAULTS = dict(max_new_tokens=384, timeout_seconds=120.0, startup_timeout_seconds=300.0)
# serve.py GOAL_REWRITER_UNUSED (84ed862): the startup record's reason for observation_query.
GOAL_REWRITER_UNUSED = 'observation_query ranks Pages by the observation vector only'
# One 784-token KV block of all attention layers, and the GDN slot (one block per Mamba group): a request's
# active KV at complete is (selected + Pages from the candidate end to its computed end + slot) blocks. Holds
# for all 60 requests of ms1 job 3291683 (v9 receipts: 51,380,224 bytes per block); a page crossing in the
# last step may still hold the previous slot (+3).
KV_BLOCK_BYTES = 51380224
GDN_SLOT_BLOCKS = 3
# Decoupled layout (DESIGN D4): one FA token over the 16 FA layers; one Mamba-pool block (one unpadded GDN layer
# state in each of the 16 Mamba tensors).
FA_TOKEN_BYTES = 65536
MAMBA_BLOCK_BYTES = 16 * 3207168
COMPACT_INT8_V1 = 'compact_int8_v1'
# Carry restore into the phase-1 GDN slot: 3 Mamba groups x 16 layers x 3,207,168 B (v9 receipts; the same bytes
# with an unpadded Mamba pool).
CARRY_H2D_BYTES = 153944064
# vLLM's hybrid alignment (platforms/interface.py) on the legacy pool; a decoupled server logs neither.
ALIGN_LOG = 'Setting attention block size to 784 tokens'
PADDING_LOG = 'Padding mamba page size by'
# WP1 (exp-decouple-wp1 platforms/interface.py _check_decoupled_block_sizes): logged after it verified the user-set
# --block-size P and --mamba-block-size M in align mode and set the Mamba page unpadded.
DECOUPLED_LOG = ('Decoupled KV layout: attention block %d tokens, Mamba block %d tokens, %d Mamba slots, unpadded '
                 'Mamba page.')
DECOUPLED_LOG_PREFIX = 'Decoupled KV layout:'
LEGACY_LAYOUT = dict(name='legacy', page_tokens=PAGE, mamba_tokens=PAGE, decoupled=False, record=None, pools=None)
# Client send/receive times against the server's queue waits: loopback transfer, JSON and thread
# scheduling stay far below this; real queue waits are seconds.
SERIAL_TOLERANCE_SECONDS = 1.0
# perf-ms16: the work outside the admission lock and outside the server's own clock (HTTP, JSON, the frontend's render
# and tokenization in the API server's event loop, the response) queues behind the other sessions' requests when they
# arrive together (16 at turn 0), so the tolerance grows by this much per other session: 1.3 s at 4 sessions, 2.5 s at
# 16. Real queue waits at 16 sessions are several seconds (16 serialized admissions).
SERIAL_TOLERANCE_PER_SESSION_SECONDS = 0.1
# Session pool: two turn-0 sends closer than this may reach the frontend's open FIFO in either order (the driver starts
# clients session.START_GAP = 0.25 s apart; a late client thread can send after the next one).
ORDER_TOLERANCE_SECONDS = 0.2


def queue_tolerance(sessions):
    """The queue-wait tolerance of a server with this many sessions."""
    return round(SERIAL_TOLERANCE_SECONDS + SERIAL_TOLERANCE_PER_SESSION_SECONDS * max(0, sessions - 1), 3)


# goal_construct_ms (on the Goal thread) lies inside goal_async_ms (parked to resumed); both are rounded to 1 us.
GOAL_ROUNDING_MS = 0.002
# A check that records a finding (not a failure): the Goal fell back.
FINDING = 'FINDING'


class CheckFailed(Exception):
    pass


class Finding(Exception):
    """A measured outcome worth reporting that is not a harness failure (the check's status is FINDING)."""
    def __init__(self, code, detail):
        super().__init__(code)
        self.code, self.detail = code, detail


def need(ok, message):
    if not ok:
        raise CheckFailed(message)


def canonical(value):
    """remote_ra_http.config.encoded (runtime digests, wire bodies)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def contract_digest(contract):
    """remote_ra_live_v2.selection_contract.digest."""
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def native_path(out, request_id):
    return Path(out) / ('native-' + hashlib.sha256(request_id.encode()).hexdigest() + '.json')


def session_ids(server_id, count):
    """remote_ra_live_v2.sessions.session_ids (test_release_parity): k = 0 keeps the server id."""
    need(type(server_id) is str and server_id and type(count) is int and count >= 1, 'server id and session count')
    return [server_id] + ['%s-s%d' % (server_id, k) for k in range(1, count)]


def route(tasks, messages):
    """remote_ra_live_v2.sessions.route (test_release_parity): the one task whose statement is in messages[1]."""
    need(type(messages) is list and len(messages) >= 2 and type(messages[1]) is dict and
         messages[1].get('role') == 'user' and type(messages[1].get('content')) is str, 'task message required')
    hits = [k for k, task in enumerate(tasks) if task['problem_statement'] in messages[1]['content']]
    need(len(hits) == 1, 'request matches %d startup tasks' % len(hits))
    return hits[0]


def ids_of(value):
    if hasattr(value, 'keys'):
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(v) for v in value]


class HFTokens:
    """Exact mode: the clean environment's Transformers tokenizer of the served model."""
    exact = True

    def __init__(self, model_dir, stop_ids=None):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        # Qwen: the tokenizer's EOS and pad; Gemma (perf-tb16's Gemma plans): its pinned stop ids (0, 1, 50, 106)
        self.stop_ids = frozenset(stop_ids) if stop_ids is not None else frozenset(
            t for t in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id) if t is not None)

    def encode(self, text):
        return ids_of(self.tokenizer(text, add_special_tokens=False))

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def render_initial(self, messages, template):
        return ids_of(self.tokenizer.apply_chat_template(messages, chat_template=template, tokenize=True,
                                                         add_generation_prompt=False, enable_thinking=False))


def _byte_decoder():
    """GPT-2 byte-level alphabet used by the Qwen ByteLevel decoder."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, extra = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + extra); extra += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class VocabDecoder:
    """Decode-only mode from tokenizer.json alone (no Transformers): offline re-verification."""
    exact = False

    def __init__(self, tokenizer_json, eos='<|im_end|>', pad='<|endoftext|>'):
        data = json.loads(Path(tokenizer_json).read_bytes())
        self.pieces = {i: t for t, i in data['model']['vocab'].items()}
        self.added = {a['id']: (a['content'], a['special']) for a in data['added_tokens']}
        names = {a['content']: a['id'] for a in data['added_tokens']}
        self.stop_ids = frozenset((names[eos], names[pad]))
        self.bytes = _byte_decoder()

    def decode(self, ids, skip_special_tokens=True):
        out = bytearray()
        for i in ids:
            if i in self.added:
                content, special = self.added[i]
                if not (special and skip_special_tokens):
                    out += content.encode('utf-8')
            else:
                out += bytes(self.bytes[c] for c in self.pieces[i])
        return out.decode('utf-8', 'replace')


class GemmaVocabDecoder:
    """Decode-only mode for Gemma 4 from its tokenizer.json alone (byte-fallback BPE: '▁' is a space, <0xNN> a byte),
    with Gemma's stop ids (pad 0, eos 1, <|tool_response> 50, <turn|> 106)."""
    exact = False
    stop_ids = frozenset((0, 1, 50, 106))

    def __init__(self, tokenizer_json):
        data = json.loads(Path(tokenizer_json).read_bytes())
        self.pieces = {i: t for t, i in data['model']['vocab'].items()}
        self.added = {a['id']: (a['content'], a['special']) for a in data['added_tokens']}

    def decode(self, ids, skip_special_tokens=True):
        out = bytearray()
        for i in ids:
            if i in self.added:
                content, special = self.added[i]
                if not (special and skip_special_tokens):
                    out += content.encode('utf-8')
                continue
            piece = self.pieces[i]
            if len(piece) == 6 and piece.startswith('<0x') and piece.endswith('>'):
                out.append(int(piece[3:5], 16))
            else:
                out += piece.replace('▁', ' ').encode('utf-8')
        return out.decode('utf-8', 'replace')


def initial_text(messages):
    """Pinned Qwen3.6 template output for plain system + user text with enable_thinking=False."""
    return ''.join('<|im_start|>%s\n%s<|im_end|>\n' % (m['role'], m['content']) for m in messages)


class Report:
    def __init__(self):
        self.checks = []

    def run(self, name, function):
        try:
            detail = function()
        except Finding as exc:
            self.checks.append(dict(check=name, status=FINDING, finding=exc.code, detail=exc.detail))
            return exc.detail
        except CheckFailed as exc:
            self.checks.append(dict(check=name, status='FAIL', detail=str(exc)))
            return None
        except Exception as exc:  # malformed or missing evidence is a failure, never a crash
            self.checks.append(dict(check=name, status='FAIL', detail=type(exc).__name__ + ': ' + str(exc),
                                    traceback=traceback.format_exc(limit=4)))
            return None
        self.checks.append(dict(check=name, status='PASS', detail=detail if detail is not None else ''))
        return detail

    @property
    def failed(self):
        return [c['check'] for c in self.checks if c['status'] not in ('PASS', FINDING)]

    @property
    def findings(self):
        return [dict(check=c['check'], finding=c['finding'], detail=c['detail']) for c in self.checks
                if c['status'] == FINDING]


def issue_sha256(task):
    return hashlib.sha256(task['problem_statement'].encode()).hexdigest()


def listed_sessions(value):
    """The session ids a startup or shutdown receipt lists (STAGE1_CONTRACT item 10), or None if absent."""
    for key in ('session_ids', 'sessions'):
        listing = value.get(key)
        if isinstance(listing, list):
            return [s.get('session_id') if isinstance(s, dict) else s for s in listing]
    return None


def splitmix64(state):
    """selection_contract.splitmix64 (780fe54; test_release_parity): (next state, 64-bit output)."""
    state = (state + _GOLDEN64) & _MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, z ^ (z >> 31)


def fill_order(eligible, seed, session_index, turn):
    """selection_contract.fill_order (780fe54; test_release_parity compares it with the sealed one): the uniformly
    random permutation of `eligible` under SplitMix64 seeded with seed * 2^32 + session * 2^16 + turn, Fisher-Yates
    from the front with rejection of the biased tail."""
    need(type(seed) is int and 0 <= seed < 1 << 32 and type(session_index) is int and 0 <= session_index < 1 << 16
         and type(turn) is int and 0 <= turn < 1 << 16, 'random fill stream key out of range')
    order, state = list(eligible), seed << 32 | session_index << 16 | turn
    for i in range(len(order) - 1):
        bound = len(order) - i
        limit = (1 << 64) - (1 << 64) % bound
        state, z = splitmix64(state)
        while z >= limit:
            state, z = splitmix64(state)
        j = i + z % bound
        order[i], order[j] = order[j], order[i]
    return order


def layout_of(expect):
    """The server's KV layout expectation (legacy when the expectation names none)."""
    return expect.get('layout') or LEGACY_LAYOUT


def recent_quota_pages(page_tokens):
    """selection_contract.default_recent_quota_pages: the upstream 64 x 16 tokens in P-token Pages."""
    return math.ceil(64 * 16 / page_tokens)


def rewriter_policies(expect):
    """The query policies whose Goal uses the rewriter on the pinned runtime (both before 84ed862)."""
    return tuple(expect.get('rewriter_policies', QUERY))


def wanted_contract(policy, expect):
    """READY's selection contract at the server's Page size: the rewriter pin only for a policy that uses it, the
    vector format `compact_int8_v1` only on a decoupled layout (the legacy contract is the 784-token one, unchanged)."""
    if policy not in SELECTIVE:
        return None
    layout = layout_of(expect)
    contract = dict(CONTRACT, page_tokens=layout['page_tokens'],
                    recent_quota_pages=recent_quota_pages(layout['page_tokens']))
    if expect.get('active_token_budget') is not None:  # serve.py --active-token-budget: the ratio is unset (E1)
        contract.update(active_ratio=None, active_token_budget=expect['active_token_budget'])
    if layout['decoupled']:
        contract['vector_format'] = COMPACT_INT8_V1
    if policy in CONTROL:  # 780fe54: only the control's contract pins the random fill; never a rewriter
        return dict(contract, goal_rewriter=None, random_fill=dict(expect.get('random_fill') or DEFAULT_RANDOM_FILL))
    if policy in rewriter_policies(expect):
        return dict(contract, goal_rewriter=dict(REWRITER_DEFAULTS, **(expect.get('rewriter') or {})))
    return dict(contract, goal_rewriter=None)


def wanted_rewriter_record(policy, expect):
    """serve.py goal_rewriter_use (84ed862): the startup record of the rewriter decision; 'skip' before it."""
    if not expect.get('rewriter_record'):
        return 'skip'
    if policy in CONTROL:  # serve.py GOAL_REWRITER_UNUSED_RANDOM_FILL: rewriter options accepted, recorded unused
        return dict(used=False, reason=GOAL_REWRITER_UNUSED_RANDOM_FILL, requested_model=None)
    if policy not in QUERY:
        return None
    used = policy in rewriter_policies(expect)
    return dict(used=used, reason=None if used else GOAL_REWRITER_UNUSED,
                requested_model=expect.get('rewriter_requested') if used else None)


def check_ready(out, policy, expect):
    ready, startup = load(out / 'ready.json'), load(out / 'startup.json')
    tasks = expect['tasks']
    need(ready.get('native_http_ready') is True and ready.get('evidence_kind') == 'NATIVE_HTTP_RUNTIME'
         and ready.get('policy') == policy and ready.get('runtime_manifest') == expect['runtime_manifest'],
         'READY identity mismatch (native_http_ready/evidence_kind/policy/runtime_manifest)')
    for name in ('input_pins_sha256', 'task_sha256', 'common_config_sha256', 'server_config_sha256'):
        need(ready.get(name) == startup.get(name), 'READY startup pin differs: ' + name)
    # serve.py: task = tasks[0]; a list longer than one adds the pinned `tasks` and `tasks_sha256`.
    need(startup.get('task') == tasks[0], 'startup task is not the sealed synthetic task 0')
    if len(tasks) > 1 or expect.get('running') is not None:
        need(startup.get('tasks') == tasks and startup.get('tasks_sha256') == digest(tasks),
             'startup task list is not the sealed %d synthetic tasks' % len(tasks))
    else:
        need('tasks' not in startup, 'a one-task server must keep the single-task startup config')
    need(ready.get('protocol') == PROTOCOL and startup.get('protocol') == PROTOCOL,
         'protocol is %r, expected %s' % (ready.get('protocol'), PROTOCOL))
    # Stage 2: one running sequence per session (serve.py, remote_ra_http.config) on the pinned KV budget.
    config = ready.get('effective_server_config')
    need(isinstance(config, dict) and config == startup.get('effective_server_config')
         and ready.get('server_config_sha256') == digest(config), 'effective server config absent or not the startup one')
    seqs = expect.get('running') or len(tasks)  # a session pool runs K sequences of its T registered tasks
    need(config.get('max_num_seqs') == seqs and config.get('max_num_batched_tokens') == MAX_BATCHED,
         'max_num_seqs %r / max_num_batched_tokens %r, expected %d / %d'
         % (config.get('max_num_seqs'), config.get('max_num_batched_tokens'), seqs, MAX_BATCHED))
    if expect.get('running') is None:
        need('session_pool' not in ready and 'max_running_sessions' not in startup,
             'a fixed-session server records a session pool')
    else:
        need(ready.get('session_pool') == dict(max_running_sessions=expect['running'], registered=len(tasks))
             and startup.get('max_running_sessions') == expect['running'],
             'READY session_pool %r / startup max_running_sessions %r, expected K = %d of %d'
             % (ready.get('session_pool'), startup.get('max_running_sessions'), expect['running'], len(tasks)))
    if expect.get('kv_cache_bytes') is not None:
        need(config.get('kv_cache_memory_bytes') == expect['kv_cache_bytes'], 'kv_cache_memory_bytes %r, expected %d'
             % (config.get('kv_cache_memory_bytes'), expect['kv_cache_bytes']))
    contract, wanted = ready.get('selection_contract'), wanted_contract(policy, expect)
    if policy in SELECTIVE:
        need(isinstance(contract, dict) and contract == startup.get('selection_contract'),
             'selection_contract absent or differs from startup.json')
        need(contract == wanted, 'selection_contract differs from the startup configuration: ' +
             ', '.join(sorted(k for k in set(contract) | set(wanted) if contract.get(k) != wanted.get(k))))
    else:
        need(contract is None and startup.get('selection_contract') is None,
             'selection_contract must be null for ' + policy)
    record = wanted_rewriter_record(policy, expect)
    if record != 'skip':
        need(startup.get('goal_rewriter') == record, 'startup goal_rewriter %r, expected %r'
             % (startup.get('goal_rewriter'), record))
    space = ready.get('vector_space')
    need(isinstance(space, dict) and space.get('kind') == 'GemmaRMSNorm', 'vector_space kind is not GemmaRMSNorm')
    attached = load(out / 'worker-attached.json')
    receipt = attached['receipt']
    need(attached['epoch'] == ready['epoch'] and attached['report']['native_ready'] is True
         and attached['report']['worker_attached'] is True
         and receipt.get('evidence_kind') == 'CUDA_PHYSICAL_TENSOR_OPERATIONS'
         and digest(receipt) == ready.get('worker_receipt_sha256'), 'worker attachment receipt mismatch')
    ids = session_ids(ready.get('session_id'), len(tasks))
    wanted = [dict(index=k, session_id=sid, instance_id=task['instance_id'], issue_sha256=issue_sha256(task))
              for k, (sid, task) in enumerate(zip(ids, tasks))]
    need(ready.get('sessions') == wanted, 'READY sessions %s, expected %s' % (ready.get('sessions'), wanted))
    need(ready.get('tasks_sha256') == startup.get('tasks_sha256'), 'READY tasks_sha256 differs from startup.json')
    need(attached.get('session_ids') == ids, 'worker-attached.json lists sessions %s, expected %s'
         % (attached.get('session_ids'), ids))
    return dict(protocol=ready['protocol'], vector_space=space, sessions=len(tasks), session_ids=ids,
                max_num_seqs=config['max_num_seqs'], kv_cache_memory_bytes=config.get('kv_cache_memory_bytes'),
                contract_sha256=contract_digest(contract) if contract else None,
                goal_rewriter=None if contract is None else contract.get('goal_rewriter') is not None)


def check_turn_http(t, request, response, previous_request, previous_content, max_tokens=MAX_TOKENS):
    need(set(request) == {'model', 'messages', 'stream'} and request['model'] == 'selective-prefix'
         and request['stream'] is False, 'request fields must be exactly model, messages, stream=false')
    messages = request['messages']
    need(len(messages) == 2 + 2 * t and all(set(m) == {'role', 'content'} and type(m['content']) is str
                                            for m in messages), 'request is not system, task, reply/observation pairs')
    if t:
        need(messages[:-2] == previous_request['messages'], 'request is not an append-only extension')
        need(messages[-2] == dict(role='assistant', content=previous_content.strip()),
             'assistant echo is not the stripped previous reply')
        need(messages[-1]['role'] == 'user' and messages[-1]['content'].startswith('<returncode>'),
             'observation must be a user <returncode> message')
    choice = response['choices'][0]
    message = choice['message']
    content = message.get('content')
    need(choice.get('finish_reason') == 'stop', 'finish_reason %r, expected stop' % choice.get('finish_reason'))
    need(not message.get('tool_calls'), 'reply carries tool_calls')
    need(type(content) is str and content.strip(), 'reply content empty')
    usage = response.get('usage') or {}
    completion = usage.get('completion_tokens')
    need(type(completion) is int and 0 < completion <= max_tokens,
         'usage.completion_tokens %r missing or out of range (reply bound %d)' % (completion, max_tokens))
    return dict(marker=MARKER in content, hit_max_tokens=completion == max_tokens, completion_tokens=completion,
                usage_prompt_tokens=usage.get('prompt_tokens'), content_chars=len(content))


def check_turn0(tokens, prompt, stable, span, messages, template):
    need(span is None and 0 < stable < len(prompt), 'turn 0 boundaries invalid')
    initial, header = prompt[:stable], prompt[stable:]
    need(tokens.decode(header, False) == HEADER, 'turn 0 does not end with the generation header')
    need(tokens.decode(initial, False) == initial_text(messages), 'turn 0 initial text differs from system + task')
    if tokens.exact:
        need(initial == tokens.render_initial(messages, template), 'turn 0 != apply_chat_template(system, task)')
        need(header == tokens.encode(HEADER), 'turn 0 header tokens differ')
    return dict(initial_tokens=len(initial))


def check_transition(tokens, prompt, following, content, completion, observation, stable, span, max_tokens=MAX_TOKENS):
    """P_{t+1} = P_t + G_t + B_t + encode(user observation) + H, split on the server's own boundaries (a reply
    that ends without EOS or the marker ends at the server's reply bound `max_tokens`)."""
    need(following[:len(prompt)] == prompt, 'prompt t+1 does not extend prompt t')
    need(type(completion) is int and completion > 0, 'usage.completion_tokens of turn t missing')
    need(span is not None and span[0] == stable and len(prompt) + completion < stable < span[1] < len(following),
         'server stable_end/observation_span inconsistent with the reply length')
    user, header = following[span[0]:span[1]], following[span[1]:]

    def split(count):
        """G_t = the reply's tokens, B_t = the upstream boundary; None if the boundary rule fails."""
        generated, boundary = following[len(prompt):len(prompt) + count], following[len(prompt) + count:stable]
        eos = bool(generated) and generated[-1] in tokens.stop_ids
        wanted = '\n' if eos else IM_END_NL
        ok = tokens.decode(boundary, False) == wanted and (not tokens.exact or boundary == tokens.encode(wanted))
        stop = 'eos' if eos else 'marker' if MARKER in content else 'length' if count == max_tokens else None
        return (generated, boundary, eos, stop) if ok and stop else None
    # usage.completion_tokens is len(G_t): vLLM v1 keeps a final EOS/stop token in output token_ids.
    counted = split(completion)
    if counted is None:
        hint = (' (a G_t with one more token ending in EOS would fit: either usage omitted a final EOS, or the '
                'generation ended without EOS/pad, marker or %d tokens)' % max_tokens) if split(completion + 1) else ''
        need(False, 'no G_t/B_t split at usage.completion_tokens=%d: boundary is not "<|im_end|>\\n" (or "\\n" after '
             'EOS/pad), or the generation ended without EOS/pad, %s or %d tokens%s'
             % (completion, MARKER, max_tokens, hint))
    generated, boundary, eos, stop = counted
    need(tokens.decode(generated) == content, 'decode(G_t) differs from the HTTP reply content')
    user_text = '<|im_start|>user\n' + observation + IM_END_NL
    need(tokens.decode(user, False) == user_text, 'observation span text differs from the user observation')
    need(tokens.decode(header, False) == HEADER, 'prompt t+1 does not end with the generation header')
    if tokens.exact:
        need(user == tokens.encode(user_text), 'observation tokens differ from encode(user message)')
        need(header == tokens.encode(HEADER), 'header tokens differ')
    return dict(stop=stop, observation_tokens=len(user), boundary_tokens=len(boundary), generated_tokens=len(generated))


def check_selection(policy, turn, row, native, contract, initial_end, page=PAGE):
    PAGE = page  # the server's Page size P (784 on the legacy pool)
    n = row['candidate_page_count']
    plan = native['plan']
    selected = list(plan['selected'])
    # A session's first request is one plain prefill: no candidate Pages (n = 0).
    need(n == (native['stable_end'] // PAGE if turn else 0) and plan['candidate_end'] == n * PAGE
         and plan['policy'] == policy, 'candidate page count/candidate_end mismatch')
    need(row['selected_logical_pages'] == selected, 'runtime selected pages differ from the native plan')
    info = dict(candidate_pages=n, selected=len(selected))
    if policy == 'full':
        need(row['query'] is None and selected == list(range(n)), 'full must restore every page')
        return info
    if policy == 'random50':
        need(row['query'] is None, 'random50 carries a query receipt')
        if n >= 2:
            need(len(selected) == n // 2 and selected == sorted(set(selected)) and all(0 <= p < n for p in selected),
                 'random50 must keep floor(n/2) distinct pages')
        else:
            need(selected == list(range(n)), 'random50 short history must keep every page')
        return info
    selection = row['query']['selection']
    body = native['query']['body']['selection']
    need(all(body.get(k) == v for k, v in selection.items()), 'runtime selection differs from the signed receipt')
    need(selection['selected_pages'] == selected, 'native plan selected != query.selection.selected_pages')
    need(selection['contract_sha256'] == contract_digest(contract), 'selection contract hash mismatch')
    need(selection['candidate_pages'] == n and selection['history_tokens'] == n * PAGE, 'selection history mismatch')
    need(body.get('initial_prompt_end') == initial_end, 'initial_prompt_end %r != turn-0 template length %r'
         % (body.get('initial_prompt_end'), initial_end))
    reason = selection['reason']
    info.update(reason=reason, capacity=selection['capacity_pages'])
    everything = list(range(n))
    if policy in CONTROL and reason is not None:
        need('fill' in selection and selection['fill'] is None, 'an inactive random_fill turn names a fill')
    if turn == 0:
        need(reason == 'cold_start' and selected == everything and selection['capacity_pages'] == n,
             'turn 0 must be cold_start with every page')
        return info
    if n * PAGE <= SWITCH:
        need(reason == 'below_switch_threshold' and selected == everything,
             'history <= %d tokens must be below_switch_threshold with every page' % SWITCH)
        return info
    mandatory = min(n, -(-initial_end // PAGE))
    # selection_contract.capacity_pages: floor(history x active_ratio), or the absolute active_token_budget (E1).
    budget = contract['active_token_budget'] if contract.get('active_token_budget') is not None else \
        math.floor(n * PAGE * contract['active_ratio'])
    capacity = min(n, max(1, mandatory, budget // PAGE))
    recent, retrieved = selection['recent_pages'], selection['retrieved_pages']
    need(reason is None, 'active selection must have reason null, got %r' % reason)
    need(selection['capacity_pages'] == capacity, 'capacity %d != expected %d' % (selection['capacity_pages'], capacity))
    need(selection['mandatory_pages'] == list(range(mandatory)), 'mandatory pages != range(M)')
    need(len(recent) <= contract['recent_quota_pages'], 'more than %d recent pages' % contract['recent_quota_pages'])
    need(set(selected) == set(selection['mandatory_pages']) | set(recent) | set(retrieved),
         'selected != mandatory U recent U retrieved')
    need(len(selected) == capacity, 'selected %d pages, capacity %d' % (len(selected), capacity))
    info.update(active=True, mandatory=len(selection['mandatory_pages']), recent=len(recent), retrieved=len(retrieved))
    if policy in CONTROL:
        info.update(check_random_fill(row, native, contract, selection, n, mandatory, capacity))
        return info
    need(selection['projection_sha256'] == PROJECTION_SHA256 and isinstance(selection['selector_config_hash'], str),
         'projection/selector config hash missing')
    return info


def check_random_fill(row, native, contract, selection, n, mandatory, capacity):
    """An active random_fill turn (780fe54; pilot-v14 pilot_checks.check_random_fill): no projection, query hash,
    score or Goal forward; the drawn Pages are exactly the first capacity - |mandatory| - |recent| Pages of
    fill_order(eligible, seed, session index, the row's turn), eligible = the candidates minus the mandatory and
    recent Pages, in draw order; the signed ranking is mandatory + recent + the whole permutation;
    query.selection.fill names the same draw."""
    body = native['query']['body']
    recent, retrieved = selection['recent_pages'], selection['retrieved_pages']
    pin = contract['random_fill']
    need(selection['projection_sha256'] is None and isinstance(selection['selector_config_hash'], str)
         and body.get('query_hash') is None and 'scores' not in body['selection'],
         'random_fill carries a projection, query hash or scores')
    need((body.get('goal') or {}).get('forward') is False and row['query'].get('query_token_sha256') is None,
         'random_fill ran a Goal forward or a query')
    eligible = sorted(set(range(n)) - set(range(mandatory)) - set(recent))
    k = row['session_index']
    order = fill_order(eligible, pin['seed'], k, row['turn'])
    drawn = order[:max(0, capacity - mandatory - len(recent))]
    fill = selection.get('fill')
    need(fill == dict(rng=pin['rng'], seed=pin['seed'], session_index=k, turn=row['turn'],
                      eligible_pages=len(eligible), drawn_pages=len(drawn)),
         'query.selection.fill %r is not the draw of seed %d, session %d, turn %d over %d eligible Pages'
         % (fill, pin['seed'], k, row['turn'], len(eligible)))
    need(retrieved == drawn, 'random_fill retrieved %s..., the seeded draw is %s...' % (retrieved[:6], drawn[:6]))
    if 'ranking' in body:
        need(body['ranking'] == list(range(mandatory)) + list(recent) + order,
             'random_fill ranking is not mandatory + recent + the permutation')
    return dict(fill_eligible=len(eligible), fill_drawn=len(drawn))


GOAL_FORWARD_OP = 'goal_forward'  # integrate-1030 --goal-overlap: a continuing turn's separate Goal forward


def frame_ops(frames, turn, arm=False, goal_overlap=False):
    """prepare, [phase1, phase1_drain on a continuing turn], reserve, allocation_update x k, complete; with `arm`
    (perf-adm: --phase1-in-step on a server whose sessions decode together) a continuing turn may run its phase 1 in a
    step: prepare, phase1_arm, phase1, phase1_drain, reserve, ...; with `goal_overlap` (integrate-1030 --goal-overlap,
    perf-swe32's rule) a continuing turn that needs a Goal forward runs it as its own frame after phase 1's drain, which
    holds its own drain: ..., phase1, phase1_drain, goal_forward, reserve, ... (which turns: tb_checks.check_goal_overlap;
    on a chat-text server a Terminus 2 summarization sub-call's turn too, its Goal built without the rewriter)."""
    ops = [f['operation'] for f in frames]
    # perf-tb16 (--release-idle-pages, idle_release.py; perf-gemma-swe32's rule): a continuing turn of a session whose
    # Pages were released readmits them before its phase 1, filed under the request as one `readmit` frame after prepare
    readmit = ['readmit'] if turn and ops[1:2] == ['readmit'] else []
    armed = arm and turn and ops[1 + len(readmit):2 + len(readmit)] == ['phase1_arm']
    head = ['prepare'] + readmit + (['phase1_arm'] if armed else []) + (['phase1', 'phase1_drain'] if turn else [])
    if goal_overlap and turn and ops[len(head):len(head) + 1] == [GOAL_FORWARD_OP]:
        head.append(GOAL_FORWARD_OP)
    head.append('reserve')
    updates = ops[len(head):-1]
    need(ops[:len(head)] == head and ops[-1:] == ['complete'] and updates
         and set(updates) == {'allocation_update'}, 'worker frame order %s' % ops)
    return len(updates)


COUNTER_KEYS = ('h2d_issued', 'h2d_joined', 'zero_joins')
D2D_KEY = 'd2d_bytes'  # perf-adm: the fourth counter of a --gdn-device-handoff server's frames


def phase2_end(stage):
    """Where a turn's phase 2 computed to: stage_b.computed_end, or on a --thinking-history strip row (perf-tb16-strip:
    computed_end is the committed cut X there) its phase2_end C'."""
    return stage.get('phase2_end', stage['computed_end'])


def own_counters(row, frames, handoff=False):
    """The worker frames carry the request's own cumulative H2D counters: zero before its first frame (never
    process totals), never decreasing, joined, and split exactly into the runtime row's prepare and restore bytes.
    perf-adm: with the hand-off every frame also carries the request's own d2d_bytes (zero before its first frame,
    never decreasing), and only then. Returns the request's H2D bytes."""
    keys = COUNTER_KEYS + ((D2D_KEY,) if handoff else ())
    first = frames[0]['http_measurement']['before']
    need({k: first.get(k) for k in keys} == dict.fromkeys(keys, 0),
         'first worker frame counters %r are not the request\'s own (zero before its first frame)'
         % {k: first.get(k) for k in keys})
    previous = first
    for frame in frames:
        before, after = frame['http_measurement']['before'], frame['http_measurement']['after']
        need(all((D2D_KEY in side) is handoff for side in (before, after)),
             '%s frame counters %s d2d_bytes (GDN hand-off %s)' % (frame['operation'], 'carry' if D2D_KEY in after
                                                                   else 'lack', 'on' if handoff else 'off'))
        need(all(type(before.get(k)) is int and type(after.get(k)) is int and previous[k] <= before[k] <= after[k]
                 for k in keys), '%s frame counters are not cumulative for this request' % frame['operation'])
        need(after['h2d_issued'] == after['h2d_joined'], 'unjoined native H2D')
        previous = after
    last = frames[-1]['http_measurement']['after']['h2d_joined']
    need(row['prepare']['h2d_bytes'] + row['restore']['h2d_bytes'] == last,
         'runtime prepare + restore H2D %d + %d != the request\'s own %d bytes'
         % (row['prepare']['h2d_bytes'], row['restore']['h2d_bytes'], last))
    return last


def own_active_kv(row, frames, layout=LEGACY_LAYOUT):
    """The complete frame's active KV is the runtime row's and counts only this request's blocks: its selected
    Pages, the Pages from the candidate end to its computed end, and its GDN slot (a crossing of the Mamba block in
    the last step may still hold the previous slot). Legacy: every block is 51,380,224 B; decoupled: FA blocks of
    P x 65,536 B and Mamba-pool blocks of 51,314,688 B. Returns the blocks."""
    active = frames[-1]['http_measurement']['active_owned_kv_bytes']
    need(row['active_kv_bytes'] == active, 'runtime active_kv_bytes differs from the complete frame')
    P = layout['page_tokens']
    base = len(row['selected_logical_pages']) + -(-phase2_end(row['stage_b']) // P) - row['candidate_page_count']
    if layout.get('window'):
        # the window layout (perf-tb16's Gemma servers): only the global layers have views; the sliding blocks are
        # claimed but never counted, so the active bytes are this request's Pages of P x 81,920 B alone
        need(active == base * layout['page_bytes'], 'active KV %d bytes is not this request\'s %d global Pages of %d B'
             % (active, base, layout['page_bytes']))
        return base
    if not layout['decoupled']:
        blocks, rest = divmod(active, KV_BLOCK_BYTES)
        need(rest == 0 and blocks - base in (GDN_SLOT_BLOCKS, 2 * GDN_SLOT_BLOCKS),
             'active KV %d bytes = %d blocks + %d, not this request\'s %d Pages + %d GDN blocks'
             % (active, blocks, rest, base, GDN_SLOT_BLOCKS))
        return blocks
    mamba, rest = divmod(active - base * P * FA_TOKEN_BYTES, MAMBA_BLOCK_BYTES)
    need(rest == 0 and mamba in (GDN_SLOT_BLOCKS, 2 * GDN_SLOT_BLOCKS),
         'active KV %d bytes is not this request\'s %d FA Pages of %d B + %d (or %d) Mamba blocks of %d B'
         % (active, base, P * FA_TOKEN_BYTES, GDN_SLOT_BLOCKS, 2 * GDN_SLOT_BLOCKS, MAMBA_BLOCK_BYTES))
    return base + mamba


def check_io(row, native, layout=LEGACY_LAYOUT, handoff=False):
    """The request's own IO and engine stall (STAGE2_CONTRACT items 3, 5 and 8).

    Batched requests share each step, so the worker frames carry the request's
    own counters and KV blocks (own_counters, own_active_kv);
    `admission_engine_ms` is that admission's wall time on the engine thread
    (the decode stall) and `capacity_waits` the row's CAPACITY_WAIT retries.
    """
    frames = native['worker_frames']
    h2d = own_counters(row, frames, handoff)
    blocks = own_active_kv(row, frames, layout)
    stall = native.get('admission_engine_ms')
    need(type(stall) in (int, float) and math.isfinite(stall) and stall > 0,
         'native admission_engine_ms missing or invalid: %r' % (stall,))
    waits = row.get('capacity_waits')
    need(type(waits) is int and waits >= 0, 'runtime capacity_waits missing or invalid: %r' % (waits,))
    return dict(h2d_bytes=h2d, active_blocks=blocks, admission_engine_ms=stall, capacity_waits=waits)


def check_stage_b(turn, row, native, previous, completion, page=PAGE):
    """Two-phase turn geometry of one session: every token is computed exactly once.

    Phase 1 computes [C, e) (the observation, on the resident Pages and the
    live tail); phase 2 computes [e, C') with C' = N + g - 1 (vLLM never
    computes the last sampled token). The session's next turn starts at C'.
    """
    PAGE = page  # the server's Page size P (784 on the legacy pool)
    stage, plan = row['stage_b'], native['plan']
    for key in ('local_turn', 'computed_start', 'sealed_start', 'candidate_end', 'observation', 'phase2_start',
                'prompt_end', 'tail_candidate', 'tail_live', 'retain', 'restore', 'release', 'run_pages'):
        need(stage[key] == plan[key], 'runtime stage_b.%s differs from the native plan' % key)
    N, span, stable = len(native['prompt_token_ids']), native['observation_span'], native['stable_end']
    C, Q, L = stage['computed_start'], stage['sealed_start'], stage['candidate_end']
    need(stage['local_turn'] == turn and native['local_turn'] == turn and stage['rebuild'] is False
         and native['rebuild'] is False, 'unexpected local turn or session rebuild')
    need(stage['prompt_end'] == N and L == (stable // PAGE * PAGE if turn else 0) and Q == C // PAGE * PAGE,
         'prompt end / candidate end / sealed start geometry')
    need(stage['tail_candidate'] == (C > Q and L == Q + PAGE) and stage['tail_live'] == (C > Q and L == Q),
         'tail flags inconsistent with C, Q, L')
    if turn == 0:
        need(C == 0 and stage['phase1'] is None and stage['phase2_start'] == 0 and stage['observation'] is None,
             'turn 0 is one plain prefill from token 0')
    else:
        phase1 = stage['phase1']
        need(C == previous['stage_b']['computed_end'], 'turn starts at %d, the session\'s previous turn computed to %d'
             % (C, previous['stage_b']['computed_end']))
        need(Q == previous['stage_b']['sealed_end'], 'sealed start differs from the session\'s previous sealed end')
        need(stage['observation'] == span and span[0] == stable, 'observation span differs from the render')
        e = span[1]
        need(phase1 is not None and phase1['start'] == C and phase1['end'] == e and phase1['forward_tokens'] == e - C
             and stage['phase2_start'] == e, 'phase 1 must compute exactly [C, e), phase 2 start at e')
    selected = set(plan['selected'])
    need(set(stage['retain']) | set(stage['restore']) == selected and not set(stage['retain']) & set(stage['restore'])
         and not set(stage['release']) & selected, 'retain/restore/release do not partition the selection')
    need(type(completion) is int and stage['computed_end'] == N + completion - 1
         and stage['sealed_end'] == stage['computed_end'] // PAGE * PAGE, "C' != N + g - 1 or Q' != floor(C'/P)P")
    return dict(C=C, Q=Q, L=L, e=span[1] if span else None, N=N, computed_end=stage['computed_end'],
                phase1_tokens=stage['phase1']['forward_tokens'] if stage['phase1'] else 0,
                phase2_tokens=stage['computed_end'] - (span[1] if span else 0),
                retain=len(stage['retain']), restore=len(stage['restore']), release=len(stage['release']),
                tail_candidate=stage['tail_candidate'], tail_live=stage['tail_live'],
                goal_forward=bool(stage['phase1'] and stage['phase1']['goal_forward']))


def failure_rows(out):
    path = Path(out) / 'failures.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def failure_summary(out):
    """First failure record with its exception chain (the engine rejection text lives there)."""
    path = out / 'failures.jsonl'
    if not path.exists() or path.stat().st_size == 0:
        return None
    first = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
    return '; '.join('%s: %s' % (c['type'], c['message']) for c in first.get('chain', ())) or \
        '%s: %s' % (first.get('error'), first.get('detail'))


def check_cleanup(out, ready, rows, ids, pooled=False, exempt=()):
    """Shutdown drains every session: both receipts name the server and list every session; all admitted requests.
    A session pool's engine receipt lists the sessions still open at shutdown (sessions.jsonl has the ones the harness
    closed: none when every task ran and was closed) and the parent receipt every registered session. perf-ctx:
    `exempt` the over-length probe's failures.jsonl rows (overlength_record): left out here (overlength-refused
    checks them), and an engine that admitted the probe before refusing it may list its internal id."""
    raw = (out / 'engine-cleanup.json').read_bytes()
    engine, parent = json.loads(raw), load(out / 'parent-cleanup.json')
    events = [json.loads(line) for line in (out / 'sessions.jsonl').read_text(encoding='utf-8').splitlines()
              if line.strip()] if pooled and (out / 'sessions.jsonl').exists() else []
    closed = {e['session_id'] for e in events if e.get('event') == 'close'}
    opened = [e['session_id'] for e in events if e.get('event') == 'open']
    still = [sid for sid in ids if sid in opened and sid not in closed] if pooled else ids
    for name, value, wanted in (('engine', engine, still), ('parent', parent, ids)):
        listing = listed_sessions(value)
        need(value.get('epoch') == ready['epoch'] and value.get('joined') is True, 'cleanup receipt epoch/join mismatch')
        need(value.get('session_id') == ready['session_id'] and listing == wanted,
             '%s cleanup names %r and lists %r, expected server %s and sessions %s'
             % (name, value.get('session_id'), listing, ready['session_id'], wanted))
    need([s.get('session_index') for s in engine['sessions']] == [ids.index(sid) for sid in still],
         'engine cleanup session indices differ from the session order')
    need(engine.get('runtime_manifest') == ready['runtime_manifest'] and engine.get('pending_operations') == 0
         and parent.get('cleanup_sha256') == hashlib.sha256(raw).hexdigest() and parent.get('natural_exit') is True
         and parent.get('child_exitcode') == 0, 'cleanup ACK/hash/natural exit mismatch')
    probe_ids = {f.get('request_id') for f in exempt}
    admitted = set(engine.get('admitted_request_ids', ()))
    wanted = {r['request_id'] for r in rows}
    need(wanted <= admitted and admitted - wanted <= probe_ids,
         'cleanup did not drain exactly the admitted requests of all sessions')
    others = [f for f in failure_rows(out) if f not in list(exempt)]
    need(not others, 'failures.jsonl is not empty: %s' % '; '.join(
        '%s: %s' % (c['type'], c['message']) for c in (others[0].get('chain') or ())) if others else None)
    return dict(admitted=len(rows), sessions=len(ids))


def check_join(rows, clients, ids):
    """Runtime rows <-> client requests: a bijection by response id, each row in its own session's turn order.

    Rows are written at completion: a session's turns appear in order, but under
    batched decode the server-wide admission order (`request_index`) is not the
    file order. Each session's request indices increase with its turns.
    """
    seen = {}
    for name in ('request_id', 'response_id'):
        values = [r.get(name) for r in rows]
        need(len(set(values)) == len(values), 'duplicate %s in runtime.jsonl' % name)
    by_response = {r['response_id']: r for r in rows}
    for k, client in enumerate(clients):
        for t, response_id in enumerate(client['response_ids']):
            need(response_id in by_response, 's%d turn %d (response %s) has no runtime row' % (k, t, response_id))
            seen[response_id] = (k, t)
    need(set(seen) == set(by_response), 'runtime rows without a client request: %s'
         % sorted(set(by_response) - set(seen))[:5])
    order, indices = {}, {}
    for row in rows:
        k, t = seen[row['response_id']]
        need(order.get(k, -1) == t - 1, 'session %d turn %d appears out of order in runtime.jsonl' % (k, t))
        order[k] = t
        index = row.get('request_index')
        need(type(index) is int and index >= 0, 's%d turn %d: request_index %r' % (k, t, index))
        need(index > indices.get(k, -1), 's%d turn %d: request_index %d does not follow the session\'s %d'
             % (k, t, index, indices.get(k, -1)))
        indices[k] = index
    values = [r['request_index'] for r in rows]
    need(len(set(values)) == len(values), 'request_index repeats: %s' % sorted(values)[:20])
    in_file = sum(1 for a, b in zip(values, values[1:]) if b < a)
    return dict(rows=len(rows), per_session=[sum(1 for v in seen.values() if v[0] == k) for k in range(len(ids))],
                request_index_gaps=max(values) + 1 - len(values) if values else 0, completed_out_of_admission_order=in_file)


def check_disjoint(out, rows, ids, ready):
    """Per-session evidence is disjoint: one session id each, own native receipts and own KV namespace.

    Without rebuilds each session keeps the namespace it opened at startup, the
    one engine-cleanup.json lists as closed for it.
    """
    sessions, namespaces = {}, {}
    for row in rows:
        need(row.get('epoch') == ready.get('epoch') and row.get('server_run_id') == ready.get('server_run_id'),
             'runtime rows of another server')
        sessions.setdefault(row.get('session_id'), []).append(row['request_id'])
        native = load(native_path(out, row['request_id']))
        namespace = (native.get('plan') or {}).get('namespace')
        if namespace is not None:
            namespaces.setdefault(row.get('session_id'), set()).add(namespace)
    need(set(sessions) <= set(ids), 'rows name sessions %s outside %s' % (sorted(set(sessions) - set(ids)), ids))
    owners = {}
    for session, values in namespaces.items():
        for namespace in values:
            need(owners.setdefault(namespace, session) == session,
                 'KV namespace %s is used by sessions %s and %s' % (namespace, owners[namespace], session))
    closed = {s.get('session_id'): s.get('namespace') for s in load(out / 'engine-cleanup.json').get('sessions') or ()}
    if (Path(out) / 'sessions.jsonl').exists():  # session pool: the harness's closes name their namespaces
        for line in (Path(out) / 'sessions.jsonl').read_text(encoding='utf-8').splitlines():
            event = json.loads(line) if line.strip() else {}
            if event.get('event') == 'close' and event.get('opened'):
                closed[event['session_id']] = event.get('namespace')
    for session, values in namespaces.items():
        need(values == {closed.get(session)}, 'session %s used namespaces %s, engine cleanup closed %r'
             % (session, sorted(values), closed.get(session)))
    return dict(sessions=len(sessions), namespaces={s: sorted(v) for s, v in sorted(namespaces.items())})


# ---------------------------------------------------------------- client-cancelled admissions (2026-10-24)
#
# Jobs 3433706 (gobsdry), 3434544 (ggoaldry) and gswe32 3429830 failed only on the admission checks below: Harbor's
# agent timeout (and the SWE pool's cap) disconnects a request while its admission waits; the frontend ends the block
# with CancelledError and compensates (frontend_compensation_start / _end of the same request_sha256: a Cancel of the
# ADD), and the ADD's own ack arrives after the block (frontend_ack_received pending false), or the engine refuses that
# ADD at the capacity gate once it reaches it. cancelled_admissions() accepts such an admission only with positive
# evidence of a clean cancellation; the checks count the clean ones (`cancelled_admissions`) and admission-log reports
# them as the finding CANCELLED_ADMISSION.

CANCELLED_OUTCOME = 'CancelledError'
DISCONNECT_TEXT = 'client disconnected'


def disconnect_failure(failure):
    """A failures.jsonl row of a request whose client disconnected ('client disconnected' / 'Client disconnected' in its
    error, detail or exception chain)."""
    parts = [str(failure.get('error')), str(failure.get('detail'))]
    parts += [str(c.get('message')) for c in failure.get('chain') or () if isinstance(c, dict)]
    return DISCONNECT_TEXT in ' '.join(parts).lower()


def cancelled_admissions(out, server_log, rows, failures, found=None, events=None):
    """The admissions whose client cancelled them while they waited (frontend_admission_end outcome CancelledError), in
    log order, each judged clean or not. Clean only with all of:
    - no runtime row (sha256 of a row's request_id) for its request_sha256;
    - the frontend compensated it: after the block exactly one frontend_compensation_start and one
      frontend_compensation_end of that request_sha256 and pid, the end's outcome ok, and the compensating command (the
      first one the pid sent after the compensation opened, with its wait) acknowledged inside the compensation;
    - its ADD (the first command the pid sent after the block opened, with the block's admission wait) acknowledged
      once, after the block with no waiter left (frontend_ack_received pending false) unless the block itself named
      it, dispatched once by the engine; a parked ADD resumed, never aborted;
    - the engine admitted it and then cancelled it: native-<request_sha256>.json names a request id that hashes to it,
      terminal_state cancelled, admitted per engine-cleanup.json; its one 'client disconnected' failure row matched by
      that request id or, when the row's request_id is null, by the receipt's session_id, session_index and turn;
    - or the engine refused that ADD at the capacity gate inside its dispatch (engine_rejected CAPACITY_WAIT of the
      key; no native receipt; no admitted request id hashes to it): the log names such a request only by its hash and
      its failure row carries no request id, so it takes one 'client disconnected' row with request_id null that no
      other cancelled admission, native receipt or runtime row claims (in log order, `matched_by`
      unclaimed_failure_row);
    - each failure row serves one admission.
    `out` is the server's output directory (native receipts, engine-cleanup.json); `found` / `events` the parsed log
    (measure.admissions / admission_events) when the caller has them. Returns dict(clean=[record], bad=[message]): a
    record names request_sha256, the engine's outcome (admitted / refused_after_cancel), request_id, session_id,
    session_index, turn (null for a refused one), matched_by, failure_index (into `failures`), add_call, parked,
    frontend_ms, compensation_ms, start_pos."""
    if found is None or events is None:
        text = Path(server_log).read_text(encoding='utf-8', errors='replace')
        found = found if found is not None else measure.admissions(text)
        events = events if events is not None else measure.admission_events(text)[0]
    out = Path(out) if out is not None else None
    failures, engine = list(failures), found['engine']
    hashes = {hashlib.sha256(r['request_id'].encode()).hexdigest() for r in rows}
    compensations, sent, acks, rejections = {}, {}, {}, {}
    for position, event in enumerate(events):
        phase = event.get('phase')
        if phase in ('frontend_compensation_start', 'frontend_compensation_end'):
            compensations.setdefault(event.get('request_sha256'), []).append((position, event))
        elif phase == 'frontend_command_sent':
            sent.setdefault(event.get('pid'), []).append((position, event))
        elif phase == 'frontend_ack_received':
            acks.setdefault(event.get('call_id'), []).append((position, event))
        elif phase == 'engine_rejected' and event.get('kind') == 'RegisteredAdd':
            rejections.setdefault(event.get('request_sha256'), []).append((position, event))
    admitted = None
    if out is not None and (out / 'engine-cleanup.json').is_file():
        admitted = {hashlib.sha256(str(r).encode()).hexdigest()
                    for r in load(out / 'engine-cleanup.json').get('admitted_request_ids') or ()}
    disconnected = [i for i, f in enumerate(failures) if disconnect_failure(f)]
    claimed, clean, bad, refused = set(), [], [], []

    def first_sent(pid, after):
        return next(((p, e) for p, e in sent.get(pid, ()) if p > after), (None, {}))

    def judge(key, value):
        need(key not in hashes, 'it has a runtime row')
        mine = compensations.get(key, [])
        need(len(mine) == 2 and [e.get('phase') for _, e in mine] == ['frontend_compensation_start',
                                                                       'frontend_compensation_end']
             and mine[0][0] > value['end_pos'] and all(e.get('pid') == value['pid'] for _, e in mine),
             'no compensation of its own after the block (%d compensation events of it)' % len(mine))
        (start, opened), (end, closed) = mine
        need(closed.get('outcome') == 'ok', 'its compensation ended %r, not ok' % (closed.get('outcome'),))
        position, cancel = first_sent(value['pid'], start)
        answered = [p for p, _ in acks.get(cancel.get('call_id'), ())]
        need(position is not None and position < end and cancel.get('timeout_seconds') == opened.get('timeout_seconds')
             and len(answered) == 1 and start < answered[0] < end,
             'its compensating command was not sent and acknowledged inside the compensation')
        position, add = first_sent(value['pid'], value['start_pos'])
        call = add.get('call_id')
        need(position is not None and position < value['end_pos'] and add.get('timeout_seconds') == value['timeout'],
             'no ADD sent inside its block')
        need(value['call_id'] in (None, call), 'the log gives its block command %s, not its ADD %s'
             % (value['call_id'], call))
        answered = acks.get(call, [])
        need(len(answered) == 1, 'its ADD %s acknowledged %d times' % (call, len(answered)))
        at, ack = answered[0]
        need((value['call_id'] == call and at < value['end_pos'])
             or (at > value['end_pos'] and ack.get('pending') is False),
             'its ADD %s was acknowledged neither inside its block nor after it with no waiter left' % call)
        dispatches = engine['calls'].get(call) or []
        need(len(dispatches) == 1, 'its ADD %s dispatched %d times' % (call, len(dispatches)))
        piece, parked = dispatches[0], engine['parked'].get(call)
        need(parked is None or (parked['resume'] is not None and parked['abort'] is None),
             'its ADD %s parked and was not resumed' % call)
        refusals = [e for p, e in rejections.get(key, ()) if piece['start_pos'] < p < piece['end_pos']]
        path = out / ('native-%s.json' % key) if out is not None else None
        native = load(path) if path is not None and path.is_file() else None
        record = dict(request_sha256=key, add_call=call, parked=parked is not None, frontend_ms=value['frontend_ms'],
                      compensation_ms=closed.get('elapsed_ms'), start_pos=value['start_pos'])
        if refusals:
            need(parked is None and all(str(e.get('error') or '').startswith(measure.CAPACITY_WAIT) for e in refusals),
                 'the engine refused its ADD for another reason: %s' % [str(e.get('error'))[:120] for e in refusals])
            need(native is None, 'a native receipt names it although the engine refused its ADD')
            need(admitted is not None and key not in admitted, 'the engine cleanup receipt does not show it unadmitted')
            record.update(engine='refused_after_cancel', request_id=None, session_id=None, session_index=None, turn=None)
            return record
        need(native is not None, 'no native receipt names it although the engine took its ADD')
        rid = native.get('request_id')
        need(isinstance(rid, str) and hashlib.sha256(rid.encode()).hexdigest() == key
             and native.get('terminal_state') == 'cancelled', 'its native receipt names request %r in state %r'
             % (rid, native.get('terminal_state')))
        need(admitted is None or key in admitted, 'the engine cleanup receipt does not list request %s' % rid)
        ident = {k: native.get(k) for k in ('session_id', 'session_index', 'turn')}
        by_id = [i for i in disconnected if failures[i].get('request_id') == rid]
        by_turn = [i for i in disconnected if failures[i].get('request_id') is None
                   and all(failures[i].get(k) == v for k, v in ident.items())]
        matches = by_id or by_turn
        need(len(matches) == 1 and matches[0] not in claimed, 'no one client-disconnected failure row of request %s '
             '(session %s turn %s): %d found' % (rid, ident['session_index'], ident['turn'], len(matches)))
        claimed.add(matches[0])
        record.update(engine='admitted', request_id=rid, matched_by='request_id' if by_id else 'session_turn',
                      failure_index=matches[0], **ident)
        return record

    cancelled = sorted(((v['start_pos'], key, v) for key, values in found['requests'].items() for v in values
                        if v['outcome'] == CANCELLED_OUTCOME and v['end_pos'] is not None), key=lambda x: x[0])
    for _, key, value in cancelled:
        try:
            record = judge(key, value)
        except CheckFailed as exc:
            bad.append('%s… (events %s-%s): %s' % ((key or '')[:12], value['start_pos'], value['end_pos'], exc))
            continue
        (refused if record['engine'] == 'refused_after_cancel' else clean).append(record)
    if refused:
        # a refused request reached the engine only under its hash: it takes a 'client disconnected' row with no
        # request id that no runtime row, native receipt or other cancelled admission accounts for
        taken = {(r.get('session_id'), r.get('turn')) for r in rows}
        if out is not None:
            for path in out.glob('native-*.json'):
                if path.name[len('native-'):-len('.json')] not in hashes:
                    native = load(path)
                    taken.add((native.get('session_id'), native.get('turn')))
        pool = [i for i in disconnected if i not in claimed and failures[i].get('request_id') is None
                and (failures[i].get('session_id'), failures[i].get('turn')) not in taken]
        for record in refused:
            if not pool:
                bad.append('%s… (event %s): the engine refused its ADD after the cancel, and no client-disconnected '
                           'failure row without a request id is left for it' % (record['request_sha256'][:12],
                                                                                record['start_pos']))
                continue
            i = pool.pop(0)
            claimed.add(i)
            record.update(matched_by='unclaimed_failure_row', failure_index=i,
                          session_id=failures[i].get('session_id'), session_index=failures[i].get('session_index'),
                          turn=failures[i].get('turn'))
            clean.append(record)
    clean.sort(key=lambda r: r['start_pos'])
    return dict(clean=clean, bad=bad)


def cancelled_summary(records):
    """The clean cancelled admissions as a check detail reports them."""
    return [dict(request_sha256=r['request_sha256'][:12], engine=r['engine'], request_id=r.get('request_id'),
                 session_index=r.get('session_index'), turn=r.get('turn'), matched_by=r.get('matched_by'),
                 parked=r['parked'], compensation_ms=r.get('compensation_ms')) for r in records]


def cancelled_parked(out, server_log, rows, found, events):
    """The parked ADDs of client-cancelled requests on a parking server (job 3434544): {call: kind}, kind
    `cancelled_admission` (a clean cancelled admission whose ADD parked at its Goal and resumed: cancelled_admissions)
    or `cancelled_request` (the accepted admission, outcome ok, of a request without a runtime row whose
    'client disconnected' failure row names its request id: admission-log's rule of job 3399075), and the number of such
    accepted admissions whose ADD the log cannot name (K > 1)."""
    failures = failure_rows(out)
    hashes = {hashlib.sha256(r['request_id'].encode()).hexdigest() for r in rows}
    calls = {r['add_call']: 'cancelled_admission' for r in cancelled_admissions(out, server_log, rows, failures, found,
                                                                                 events)['clean'] if r['parked']}
    unnamed = 0
    for failure in failures:
        rid = failure.get('request_id')
        if not isinstance(rid, str) or not rid or not disconnect_failure(failure):
            continue
        key = hashlib.sha256(rid.encode()).hexdigest()
        accepted = [v for v in measure.accepted_admissions(found['requests'].get(key, [])) if v['outcome'] == 'ok']
        if key in hashes or len(accepted) != 1:
            continue
        if accepted[0]['add_ambiguous']:
            unnamed += 1
        elif accepted[0]['call_id'] in found['engine']['parked']:
            calls[accepted[0]['call_id']] = 'cancelled_request'
    return calls, unnamed


def check_admission_log(server_log, rows, failures=(), exempt=frozenset(), out=None):
    """Serialized admissions and their CAPACITY_WAIT refusals, as the server log shows them (STAGE2_CONTRACT 2, 3, 5).

    The admission lock passes on at engine acceptance, so admission blocks
    (frontend_admission_start .. frontend_admission_end) never overlap: no
    admission starts before the previous one's ack (which arrives inside its
    block). Each runtime row has exactly one accepted admission; every other
    admission is a capacity-gate refusal, and their number is the rows' (and
    failures') `capacity_waits`. No admission was rejected for another reason.
    In log order, the accepted admissions of rows without capacity waits follow
    their request_index (first admission, FIFO).

    goal_async (ms2a): an admission parked at its Goal is the union of its
    engine-thread pieces, its dispatch (up to the park) and its resume (up to the
    ack). The engine thread's pieces, of admissions and of the completions'
    commands alike, never overlap (measure.engine_pieces `overlaps`); the parked
    interval between them may hold decode steps and other requests' completion
    commands, and still lies inside the admission's frontend block.

    perf-ctx: `exempt` the request_sha256 keys of the over-length probe (overlength_record): an engine that refused it
    shows one rejected admission, which overlength-refused checks.

    2026-10-24 (jobs 3433706, 3434544, gswe32 3429830): an admission its client cancelled while it waited ends
    CancelledError, its ADD acknowledged after the block. It passes only as a clean cancellation (cancelled_admissions:
    compensated ok, no runtime row, its 'client disconnected' failure row, the engine's own record of it; `out` the
    server's output directory with the native receipts); every other CancelledError admission fails. The clean ones
    are counted (`cancelled_admissions`) and reported as the finding CANCELLED_ADMISSION.
    """
    text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    found = measure.admissions(text)
    overlaps = found['engine']['overlaps']
    need(not overlaps, 'engine-thread admission pieces overlap: %s' % overlaps[:3])
    K = found['permits']
    need(K is not None, 'admission blocks wait %s s for their ACK: no pinned --admission-overlap K gives that'
         % found['admission_timeouts'])
    values = sorted((v['start_pos'], key, v) for key, items in found['requests'].items() for v in items)
    need(values, 'no admission in the server log')
    failures = list(failures)
    cancelled = cancelled_admissions(out, server_log, rows, failures, found, measure.admission_events(text)[0])
    need(not cancelled['bad'], 'client-cancelled admissions without a clean cancellation: admission %s'
         % '; admission '.join(cancelled['bad'][:3]))
    clean = {r['start_pos']: r for r in cancelled['clean']}
    for (_, key, value) in values:
        if key in exempt:
            continue
        need(value['end_pos'] is not None, 'admission %s… never ended' % (key or '')[:12])
        need(value['ack_pos'] is not None or (K > 1 and value['add_ambiguous']) or value['start_pos'] in clean,
             'admission %s… has no acknowledged command inside its block' % (key or '')[:12])
        need(value['rejected'] is None, 'admission %s… rejected by the engine: %s' % ((key or '')[:12], value['rejected']))
    if K == 1:
        for (_, a, previous), (_, b, current) in zip(values, values[1:]):
            need(current['start_pos'] > previous['end_pos'], 'admission %s… started before admission %s… was '
                 'acknowledged (events %d < %d)' % ((b or '')[:12], (a or '')[:12], current['start_pos'],
                                                    previous['end_pos']))
    widest = 1
    for i, (start, b, current) in enumerate(values):  # (f): at most K admissions in flight
        open_ = 1 + sum(1 for _, _, earlier in values[:i] if earlier['end_pos'] > start)
        widest = max(widest, open_)
        need(open_ <= K, 'admission %s… started with %d admissions in flight, K = %d' % ((b or '')[:12], open_, K))
    hashes = {hashlib.sha256(r['request_id'].encode()).hexdigest(): r for r in rows}
    for key, row in hashes.items():
        accepted = measure.accepted_admissions(found['requests'].get(key, []))
        need(len(accepted) == 1 and accepted[0]['outcome'] == 'ok', 'request %s… has %d accepted admissions, outcome %r'
             % (key[:12], len(accepted), accepted and accepted[0]['outcome']))
    refused = [(key, v) for _, key, v in values if v['capacity_wait']]
    accepted = [(key, v) for _, key, v in values if not v['capacity_wait'] and key not in exempt]
    # perf-tb16 (job 3399075): an admitted request its client cancelled (Harbor's agent timeout disconnects it) has
    # no runtime row but its failures.jsonl row, under the engine request id (evidence.failed: call.internal_id)
    failed = {hashlib.sha256(f['request_id'].encode()).hexdigest(): f for f in failures
              if isinstance(f.get('request_id'), str) and f['request_id']}
    stray = [key for key, v in accepted if key not in hashes and key not in failed and v['start_pos'] not in clean]
    need(not stray, 'accepted admissions without a runtime row or a failure row: %s' % [k[:12] for k in stray[:5]])
    counted = sum(r.get('capacity_waits') or 0 for r in rows) + sum(f.get('capacity_waits') or 0 for f in failures)
    need(len(refused) == counted, '%d CAPACITY_WAIT refusals in the log, %d capacity waits in the rows'
         % (len(refused), counted))
    # the cancelled requests keep their admission's request_index (failure rows; a clean cancelled admission the engine
    # admitted: the failure row matched to its native receipt)
    indexed = dict(failed, **{r['request_sha256']: failures[r['failure_index']] for r in clean.values()
                              if r['engine'] == 'admitted'})
    indexed.update(hashes)
    plain = [indexed[key]['request_index'] for key, v in accepted
             if key in indexed and indexed[key].get('capacity_waits') == 0 and type(indexed[key].get('request_index')) is int]
    if K == 1:
        need(all(a < b for a, b in zip(plain, plain[1:])), 'admission order differs from request_index: %s' % plain[:20])
    else:  # (f): K admissions proceed side by side, so their blocks may open in another order within the K in flight
        rank = {v: i for i, v in enumerate(sorted(plain))}
        need(all(abs(rank[v] - i) < K for i, v in enumerate(plain)),
             'admission order departs from request_index by K = %d or more: %s' % (K, plain[:20]))
    pieces = found['engine']['pieces']
    detail = dict(admissions=len(values), accepted=len(accepted), capacity_wait_refusals=len(refused), permits=K,
                  in_flight_max=widest,
                  add_ambiguous=sum(1 for _, _, v in values if v['add_ambiguous']), unreadable=found['unreadable'],
                  engine_pieces=len(pieces), resume_pieces=sum(1 for p in pieces if p['kind'] == 'engine_resume'),
                  cancelled_admissions=len(clean))
    if clean:
        raise Finding('CANCELLED_ADMISSION', dict(detail, cancelled=cancelled_summary(cancelled['clean'])))
    return detail


def check_goal_async(out, server_log, rows, policy, expect):
    """Goals built off the engine thread (goal_async, runtime 9b18421), as the server log and the native records show.

    Every parked admission has exactly one resume and one ack, in order: its dispatch ends at the park, a later
    resume piece holds its one engine_ack_queued (matched by call_id), and nothing else acknowledges it
    (measure.engine_pieces `errors`); none was aborted at its Goal. On a parking server (`parking_policies`:
    goal_query with N > 1 and the rewriter) every accepted admission parked, and exactly those; their native
    records carry goal_async_ms > 0 and goal_construct_ms <= goal_async_ms. No other server parks, and its
    native records carry neither field. 2026-10-24 (job 3434544): a client-cancelled request's admission may have
    parked too (cancelled_parked: a clean cancelled admission, or the accepted admission of a request whose
    'client disconnected' failure row names it); those parked calls are counted (`cancelled_admissions`,
    `cancelled_requests`), any other one still fails.
    """
    out = Path(out)
    text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    found = measure.admissions(text)
    engine = found['engine']
    need(not engine['errors'], 'engine acknowledgements: %s' % engine['errors'][:3])
    # --deferred-completion-io (120bc3c): a completion's Cancel may be acknowledged once its files are written, later
    # than its dispatch (measure.engine_pieces `held`); any other server acknowledges every command at once
    held = engine.get('held') or {}
    need(not held or expect.get('deferred_completion_io'), '%d completion commands acknowledged after their dispatch '
         '(held) on a server without --deferred-completion-io: calls %s' % (len(held), sorted(held)[:3]))
    K = overlap_permits(expect)
    need(found['permits'] == K, 'the admission blocks wait %s s for their ACK (K = %s), the run pins '
         '--admission-overlap %d' % (found['admission_timeouts'], found['permits'], K))
    parked = engine['parked']
    aborted = [call for call, entry in parked.items() if entry['abort'] is not None]
    need(not aborted, '%d admissions failed while parked at their Goal (calls %s)' % (len(aborted), aborted[:3]))
    parks = policy in (expect.get('parking_policies') or ())
    in_step = phase1_injector(expect)  # perf-adm: an in-step phase 1 parks its admission too
    leases = K > 1 and in_step  # (f) with (b): an admission may park for the one phase-1 lease
    hashes = {hashlib.sha256(r['request_id'].encode()).hexdigest(): r for r in rows}
    calls, timings, unnamed, waits = set(), [], [], []
    for key, row in hashes.items():
        accepted = measure.accepted_admissions(found['requests'].get(key, []))
        need(len(accepted) == 1, 'request %s… has %d accepted admissions' % (key[:12], len(accepted)))
        value = accepted[0]
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        native = load(native_path(out, row['request_id']))
        fields = {k: native.get(k) for k in ('goal_async_ms', 'goal_construct_ms')}
        if 'lease_wait_ms' in native:
            wait = native['lease_wait_ms']
            need(leases, '%s: native lease_wait_ms on a server without --admission-overlap and --phase1-in-step'
                 % where)
            need(type(wait) in (int, float) and math.isfinite(wait) and wait >= 0,
                 '%s: native lease_wait_ms invalid: %r' % (where, wait))
            waits.append(wait)
        if K > 1 and value['add_ambiguous']:
            unnamed.append(where)  # (f): the log cannot name its ADD among the admissions in flight
            continue
        if parks:
            need(value['parked'] is True, '%s: its admission did not park at its Goal (ADD %s)'
                 % (where, 'unnamed' if value['call_id'] is None else value['call_id']))
            calls.add(value['call_id'])
            waited, built = fields['goal_async_ms'], fields['goal_construct_ms']
            need(type(waited) in (int, float) and math.isfinite(waited) and waited > 0,
                 '%s: native goal_async_ms missing or invalid: %r' % (where, waited))
            need(type(built) in (int, float) and math.isfinite(built) and 0 <= built <= waited + GOAL_ROUNDING_MS,
                 '%s: native goal_construct_ms %r outside [0, goal_async_ms %r]' % (where, built, waited))
            timings.append((waited, built, value['dispatch_ms'], value['resume_ms']))
        else:
            phase1 = in_step and ('phase1_park_ms' in native or 'lease_wait_ms' in native)
            need(bool(value['parked']) is phase1, '%s %s an admission (%s): its native record %s phase1_park_ms'
                 % (policy, 'parked' if value['parked'] else 'did not park', where,
                    'carries' if 'phase1_park_ms' in native else 'lacks'))
            if phase1:
                calls.add(value['call_id'])
            need(not any(k in native for k in fields), '%s: native record of %s carries %s'
                 % (policy, where, sorted(k for k in fields if k in native)))
    cancelled, cancelled_unnamed = {}, 0
    if parks or in_step:
        cancelled, cancelled_unnamed = cancelled_parked(out, server_log, rows, found, measure.admission_events(text)[0])
        others = set(parked) - set(cancelled)
        if unnamed or cancelled_unnamed:  # (f): every named admission parked; the unnamed ones are counted, not matched
            need(calls <= set(parked) and len(others) <= len(calls) + len(unnamed) + cancelled_unnamed,
                 '%d parked calls, %d of them the rows\' named admissions, %d unnamed, %d of client-cancelled requests'
                 % (len(parked), len(set(parked) & calls), len(unnamed) + cancelled_unnamed,
                    len(set(parked) & set(cancelled))))
        else:
            need(others == calls, '%d parked calls, %d of them the rows\' admissions, %d of client-cancelled requests'
                 % (len(parked), len(set(parked) & calls), len(set(parked) & set(cancelled))))
    else:
        resumes = sum(1 for p in engine['pieces'] if p['kind'] == 'engine_resume')
        need(not parked and not resumes, '%s parked %d admissions (%d resume pieces)' % (policy, len(parked), resumes))
    refusals = sum(1 for items in found['requests'].values() for v in items if v['parked_refusal'])
    top = lambda index: max([t[index] for t in timings if t[index] is not None], default=None)
    kinds = [cancelled[c] for c in set(parked) & set(cancelled)]
    return dict(parks=parks, phase1_parks=in_step, parked=len(parked), permits=K, held_acks=len(held),
                cancelled_admissions=kinds.count('cancelled_admission'),
                cancelled_requests=kinds.count('cancelled_request'),
                unattributed_resumes=len(engine.get('unattributed') or ()), unnamed_admissions=unnamed,
                lease_waits=len(waits), lease_wait_ms_max=max(waits, default=None),
                reparked=sum(1 for e in parked.values() if e.get('reparks')),
                resumed=sum(1 for e in parked.values() if e['resume'] is not None),
                refused_while_parked=refusals, goal_async_ms_max=top(0), goal_construct_ms_max=top(1),
                dispatch_ms_max=top(2), resume_ms_max=top(3),
                goal_async_s=round(sum(t[0] for t in timings) / 1000, 3) if timings else 0.0)


def check_goal_order(out, rows, ids):
    """(f) per-session Goal order (goal_query): each session's native records name that session and its turns 0, 1,
    ... in order, each with a Goal record (query.body.goal and goal_hash); with several admissions in flight each
    session still has at most one admission and builds its Goals on its own thread in its own order. Returns the
    applied decisions per session."""
    out, by = Path(out), {}
    for row in rows:
        native = load(native_path(out, row['request_id']))
        body = (native.get('query') or {}).get('body') or {}
        by.setdefault(row.get('session_index'), []).append((row.get('turn'), native, body))
    decisions = {}
    for k, items in sorted(by.items(), key=lambda item: item[0]):
        items.sort(key=lambda item: item[0])
        turns = [t for t, _, _ in items]
        need(turns == list(range(len(turns))), 'session %s: turns %s are not 0..%d in order' % (k, turns, len(turns) - 1))
        for t, native, body in items:
            need(native.get('session_index') == k and native.get('turn') == t,
                 'session %s turn %s: its native record names session %r turn %r' % (k, t, native.get('session_index'),
                                                                                    native.get('turn')))
            own = ids[k] if ids and type(k) is int and k < len(ids) else None
            need(own is None or 'session_id' not in native or native['session_id'] == own,
                 'session %s turn %s: its native record belongs to session %r' % (k, t, native.get('session_id')))
            need(isinstance(body.get('goal'), dict) and body.get('goal_hash'),
                 'session %s turn %s: no Goal record (query.body.goal / goal_hash)' % (k, t))
        decisions[str(k)] = [body['goal'].get('applied_decision') for _, _, body in items]
    return dict(sessions=len(by), decisions=decisions)


def check_goal_fallback(out, rows, policy, expect):
    """GOAL_FALLBACK (a finding, never a failure): a turn whose Goal fell back (query.body.goal.fallback_reason of
    the native record not null; for a policy without the rewriter, other than its by-design `rewriter_disabled`).
    With the rewriter that is a timeout (120 s, possibly from sharing the GPU with decode) or a failed rewrite."""
    out, fallbacks, turns = Path(out), [], 0
    allowed = (None,) if policy in rewriter_policies(expect) else (None, 'rewriter_disabled')
    for row in sorted(rows, key=lambda r: (r.get('session_index'), r.get('turn'))):
        native = load(native_path(out, row['request_id']))
        goal = ((native.get('query') or {}).get('body') or {}).get('goal')
        need(isinstance(goal, dict) and 'fallback_reason' in goal, 's%s turn %s: native query.body.goal has no '
             'fallback_reason' % (row.get('session_index'), row.get('turn')))
        turns += 1
        if goal['fallback_reason'] not in allowed:
            fallbacks.append(dict(session=row.get('session_index'), turn=row.get('turn'), reason=goal['fallback_reason'],
                                  gate_decision=goal.get('gate_decision'), applied_decision=goal.get('applied_decision'),
                                  slm_attempted=goal.get('slm_attempted'), slm_succeeded=goal.get('slm_succeeded'),
                                  goal_async_ms=native.get('goal_async_ms')))
    detail = dict(turns=turns, fallbacks=fallbacks)
    if fallbacks:
        raise Finding('GOAL_FALLBACK', detail)
    return detail


def check_queue_waits(rows, timeline, tolerance=None):
    """Queue waits consistent with FIFO admission and batched service, on the client's clock.

    `queue_wait_s` is the wait for the admission lock plus every CAPACITY_WAIT
    retry; `request_latency_ns` runs from the first admission to completion, so
    it includes the retries as well. Both lie within the client's latency; the
    client latency minus the server latency is the first lock wait, so the
    queue wait is at least that, and with no retry exactly that. FIFO: in
    request_index order (first admission) the requests were sent in order.
    """
    tolerance = SERIAL_TOLERANCE_SECONDS if tolerance is None else tolerance
    previous, waits, retried = None, [], 0
    for row in sorted(rows, key=lambda r: r.get('request_index') if type(r.get('request_index')) is int else -1):
        k, t, send, recv = timeline[row['response_id']]
        # Session pool: the wait for an open session comes first; the request reaches the admission lock after it.
        opened = row.get('open_wait_s') or 0.0
        send = send + opened if type(opened) in (int, float) and math.isfinite(opened) else send
        wait, retries = row.get('queue_wait_s'), row.get('capacity_waits')
        need(type(wait) in (int, float) and math.isfinite(wait) and wait >= 0,
             's%d turn %d: queue_wait_s missing or invalid: %r' % (k, t, wait))
        need(type(retries) is int and retries >= 0, 's%d turn %d: capacity_waits missing or invalid: %r' % (k, t, retries))
        served = row.get('request_latency_ns', 0) / 1e9
        need(send + wait <= recv + tolerance and served <= recv - send + tolerance,
             's%d turn %d: queue wait %.3f s or server latency %.3f s exceed the client latency %.3f s'
             % (k, t, wait, served, recv - send))
        need(wait >= recv - send - served - tolerance, 's%d turn %d: queue wait %.3f s is shorter than the %.3f s '
             'before its first admission (client latency %.3f s - server latency %.3f s)'
             % (k, t, wait, recv - send - served, recv - send, served))
        if retries == 0:
            need(send + wait + served <= recv + tolerance, 's%d turn %d: queue wait %.3f s + server latency %.3f s '
                 'exceed the client latency %.3f s' % (k, t, wait, served, recv - send))
        else:
            retried += 1
        if previous is not None:
            need(send >= previous[2] - tolerance, 's%d turn %d was admitted before s%d turn %d, which was sent earlier'
                 % (previous[0], previous[1], k, t))
        previous = (k, t, send, recv)
        waits.append(wait)
    return dict(requests=len(waits), queued=sum(1 for w in waits if w > tolerance), retried=retried,
                max_queue_wait_s=max(waits) if waits else None, tolerance_s=tolerance)


def check_no_preemption(out):
    """STAGE2_CONTRACT item 5: vLLM's preemption counter is 0 in every /metrics scrape of the server (chain and
    batch samples), and the server's last scrape (the final sample, else the chain scrape after the last response)
    shows it, so no preemption can hide after it."""
    out = Path(out)
    chain = measure.load_chain(out)
    samples = measure.load_samples(out)
    texts = [(r['file'], t) for r, t in zip(chain, measure.chain_texts(out, chain))]
    texts += [(r['file'], (out / r['file']).read_text(encoding='utf-8')) for r in samples]
    seen, nonzero, shown = 0, [], {}
    for name, text in texts:
        family = measure.snapshot(text).get(measure.PREEMPTIONS) if text else None
        shown[name] = family is not None
        if family is None:
            continue
        seen += 1
        total = sum(v for v in family['series'].values() if measure.finite(v) is not None)
        if total:
            nonzero.append((name, total))
    need(seen, 'no /metrics scrape shows %s: preemption cannot be ruled out' % measure.PREEMPTIONS)
    finals = [r['file'] for r in samples[-1:]] + [r['file'] for r in chain[-1:] if r['session'] is not None]
    need(any(shown.get(name) for name in finals), 'neither the last sample nor the scrape after the last response '
         'shows %s' % measure.PREEMPTIONS)
    need(not nonzero, 'vLLM preempted: %s' % nonzero[:3])
    return dict(scrapes=seen, chain=len(chain), samples=len(samples))


def load_client(out, k, n):
    sdir = out / ('s%d' % k)
    if (sdir / 'session.json').exists():
        return load(sdir / 'session.json')
    return dict(planned_turns=None, completed_turns=0, stop_reason='session_record_missing',
                error='s%d/session.json missing' % k, turns=[])


def check_goal_rewriter(policy, goals, expect, stops):
    """goal-rewriter-attempted (a policy that uses it) or goal-rewriter-unused (observation_query after 84ed862)."""
    if policy in rewriter_policies(expect):
        need(any(g and g.get('slm_attempted') for g in goals), 'query.goal.slm_attempted never true')
        return [dict(slm_attempted=g.get('slm_attempted'), slm_succeeded=g.get('slm_succeeded'),
                     fallback_reason=g.get('fallback_reason')) for g in goals]
    need(goals and all(isinstance(g, dict) for g in goals), 'query rows lack their Goal record')
    attempted = [i for i, g in enumerate(goals) if g.get('slm_attempted') is not False or g.get('slm_succeeded') is not False]
    need(not attempted, '%s attempted the Goal rewriter on %d turns' % (policy, len(attempted)))
    rewrites = [g for g in goals if g.get('gate_decision') == 'REWRITE']
    need(all(g.get('fallback_reason') == 'rewriter_disabled' for g in rewrites),
         'a REWRITE gate without the rewriter must fall back with rewriter_disabled: %s'
         % sorted({str(g.get('fallback_reason')) for g in rewrites}))
    return dict(turns=len(goals), rewrite_gates=len(rewrites))


def check_policy(out, policy, expect, tokens, server_log):
    """All checks of one server with N sessions; returns dict(status, checks, failed, turns, sessions, ...)."""
    out = Path(out)
    report = Report()
    tasks = expect['tasks']
    n = len(tasks)
    layout = layout_of(expect)
    P = layout['page_tokens']
    reply_bound = expect.get('max_tokens') or MAX_TOKENS  # perf-ctx: the server's pinned reply bound
    report.run('ready', lambda: check_ready(out, policy, expect))
    report.run('kv-layout', lambda: check_kv_layout(out, expect))
    report.run('startup-log', lambda: check_startup_log(server_log, expect))
    report.run('cuda-graphs', lambda: check_cuda_graphs(out, expect, server_log))
    report.run('admission-flags', lambda: check_admission_flags(out, expect))
    handoff, arm = gdn_handoff(expect), phase1_injector(expect)
    if expect.get('capture_sizes') and layout.get('cuda_graphs', 'off') != 'off':  # perf-ctx: the pinned sizes
        report.run('capture-sizes', lambda: check_capture_sizes(out, expect, server_log))
    ready_json = load(out / 'ready.json') if (out / 'ready.json').exists() else {}
    try:
        ids = session_ids(ready_json.get('session_id'), n)
    except CheckFailed:
        ids = [None] * n
    rows = measure.runtime_rows(out)
    template = (out / 'chat_template.jinja').read_text(encoding='utf-8') if (out / 'chat_template.jinja').exists() else None
    contract = ready_json.get('selection_contract')
    by_response = {}
    for row in rows:
        by_response.setdefault(row.get('response_id'), row)
    clients, turns, timeline, active_turns, goals = [], [], {}, 0, []
    for k in range(n):
        client = load_client(out, k, n)
        sdir = out / ('s%d' % k)
        completed = client['completed_turns']
        requests, responses, natives, prompts, contents, completions, session_rows = [], [], [], [], [], [], []
        for t in range(completed):
            requests.append(load(sdir / ('http-request-%d.json' % t)))
            responses.append(load(sdir / ('http-response-%d.json' % t)))
            row = by_response.get(responses[-1].get('id'))
            session_rows.append(row)
            path = native_path(out, row['request_id']) if row else None
            natives.append(load(path) if path and path.exists() else {})
            prompts.append(natives[-1].get('prompt_token_ids', []))
            message = responses[-1]['choices'][0]['message']
            contents.append(message.get('content') if type(message.get('content')) is str else '')
            completions.append((responses[-1].get('usage') or {}).get('completion_tokens'))
            entry = client['turns'][t] if t < len(client['turns']) else {}
            timeline[responses[-1].get('id')] = (k, t, entry.get('t_send_s'), entry.get('t_recv_s'))
        clients.append(dict(client, response_ids=[r.get('id') for r in responses]))

        def session_check(k=k, client=client, session_rows=session_rows):
            need(client['error'] is None and client['stop_reason'] in STOPS_OK,
                 'session %d ended with %s: %s' % (k, client['stop_reason'], client['error']))
            mine = [r for r in rows if r.get('session_id') == ids[k]]
            need(len(mine) == client['completed_turns'] and all(r is not None for r in session_rows),
                 'session %d: %d runtime rows for %d completed turns' % (k, len(mine), client['completed_turns']))
            return dict(planned=client['planned_turns'], completed=client['completed_turns'],
                        stop_reason=client['stop_reason'], session_id=ids[k])
        report.run('s%d-session' % k, session_check)
        initial_end = natives[0].get('stable_end') if natives else None
        for t in range(len(natives)):
            row, native, request, response = session_rows[t], natives[t], requests[t], responses[t]
            name = lambda kind, t=t: measure.check_name(k, t, kind)

            def runtime(t=t, row=row, native=native, request=request, response=response):
                need(row is not None, 'no runtime row for response %s' % response.get('id'))
                need(row['response_id'] == response['id'] and row['policy'] == policy,
                     'runtime row does not match the HTTP response/policy')
                need(row.get('session_id') == ids[k] and row.get('session_index') == k and row.get('turn') == t
                     and row.get('ordinal') == t + 1, 'runtime row names session %r index %r turn %r ordinal %r, '
                     'expected %s, %d, %d, %d' % (row.get('session_id'), row.get('session_index'), row.get('turn'),
                                                  row.get('ordinal'), ids[k], k, t, t + 1))
                need(type(row.get('request_index')) is int, 'request_index missing')
                need(route(tasks, request['messages']) == k, 'the task message routes to another session')
                wait = row.get('queue_wait_s')
                need(type(wait) in (int, float) and math.isfinite(wait) and wait >= 0, 'queue_wait_s missing or invalid')
                for key in ('epoch', 'server_run_id'):
                    need(row[key] == ready_json.get(key), 'runtime identity mismatch: ' + key)
                for field in ('prepare', 'restore', 'purge', 'drain'):
                    need(row[field]['joined'] is True, 'unjoined ' + field)
                need(row['drain']['pending_operations'] == 0 and row['fallback'] is False, 'pending/fallback runtime')
                path = native_path(out, row['request_id'])
                need(sha_file(path) == row['native_evidence_sha256'], 'native receipt hash mismatch')
                need(native['request_id'] == row['request_id'] and native['turn'] == t
                     and native['session_id'] == ids[k] and native['terminal_state'] == 'complete'
                     and native['drain']['drained'] is True, 'native completion/session/turn/drain mismatch')
                need(native.get('session_index') == k, 'native session_index %r' % native.get('session_index'))
                need(native['prompt_token_ids'] == row['prompt_token_ids']
                     and row['prompt_token_sha256'] == digest(row['prompt_token_ids']), 'prompt token identity mismatch')
                need(row['wire_sha256'] == hashlib.sha256(canonical(request)).hexdigest(), 'request wire hash mismatch')
                frames = native['worker_frames']
                need(frames and frames[-1]['operation'] == 'complete', 'missing worker completion frame')
                frame_ops(frames, t, arm)
                if policy in SELECTIVE:
                    need(row['query'].get('issue_sha256') == issue_sha256(tasks[k]), "query row names another session's issue")
                if policy in SELECTIVE and t:
                    span, linked = native['observation_span'], row['query']['observation']
                    need([linked['start'], linked['end']] == span and linked['message_indices'] == [len(request['messages']) - 1],
                         'observation linkage mismatch')
                return None
            report.run(name('runtime'), runtime)
            report.run(name('io'), lambda row=row, native=native: check_io(row, native, layout, handoff))
            http = report.run(name('http'), lambda t=t: check_turn_http(
                t, requests[t], responses[t], requests[t - 1] if t else None, contents[t - 1] if t else None,
                reply_bound))
            if t == 0:
                stream = report.run(name('token-stream'), lambda: check_turn0(
                    tokens, prompts[0], natives[0]['stable_end'], natives[0]['observation_span'],
                    requests[0]['messages'], template))
            else:
                stream = report.run(name('token-stream'), lambda t=t: check_transition(
                    tokens, prompts[t - 1], prompts[t], contents[t - 1], completions[t - 1],
                    requests[t]['messages'][-1]['content'], natives[t]['stable_end'], natives[t]['observation_span'],
                    reply_bound))
            stage = report.run(name('stage-b'), lambda t=t: check_stage_b(
                t, session_rows[t], natives[t], session_rows[t - 1] if t else None, completions[t], P))
            selection = report.run(name('selection'), lambda t=t, initial_end=initial_end: check_selection(
                policy, t, session_rows[t], natives[t], contract, initial_end, P))
            if selection and selection.get('active'):
                active_turns += 1
            if policy in SELECTIVE and isinstance(row, dict) and isinstance(row.get('query'), dict):
                goals.append(row['query'].get('goal'))
            entry = client['turns'][t] if t < len(client['turns']) else {}
            turns.append(dict(session=k, turn=t, session_id=ids[k], prompt_tokens=len(prompts[t]),
                              stable_end=native.get('stable_end'),
                              server_latency_s=round((row or {}).get('request_latency_ns', 0) / 1e9, 3),
                              queue_wait_s=(row or {}).get('queue_wait_s'),
                              capacity_waits=(row or {}).get('capacity_waits'),
                              admission_engine_ms=native.get('admission_engine_ms'),
                              goal_async_ms=native.get('goal_async_ms'), goal_construct_ms=native.get('goal_construct_ms'),
                              client_latency_s=entry.get('latency_s'),
                              t_send_s=entry.get('t_send_s'), t_recv_s=entry.get('t_recv_s'),
                              prepare_h2d_bytes=(row or {}).get('prepare', {}).get('h2d_bytes'),
                              restore_h2d_bytes=(row or {}).get('restore', {}).get('h2d_bytes'),
                              restore_d2d_bytes=(row or {}).get('restore', {}).get('d2d_bytes'),
                              phase1_mode=native.get('phase1_mode'), phase1_park_ms=native.get('phase1_park_ms'),
                              phase1_step_ns=native.get('phase1_step_ns'), lease_wait_ms=native.get('lease_wait_ms'),
                              active_kv_bytes=(row or {}).get('active_kv_bytes'),
                              kv_pools=(row or {}).get('kv_pools'), http=http, token_stream=stream,
                              stage_b=stage, selection=selection,
                              goal=row['query'].get('goal') if isinstance(row, dict) and isinstance(row.get('query'), dict)
                              else None))
    failures = failure_rows(out)
    # perf-ctx: the over-length probe's own failure row and engine admission (overlength-refused checks them).
    _, probe_failures, probe_keys = overlength_record(out, clients, expect)
    report.run('context-limits', lambda: check_context_limits(out, expect, server_log, turns))
    if expect.get('context'):
        report.run('long-reply', lambda: check_long_reply(out, expect, clients, turns))
        report.run('overlength-refused', lambda: check_overlength(out, expect, clients, tokens, server_log,
                                                                  last_turn_tokens(out, rows)))
    report.run('evidence-join', lambda: check_join(rows, clients, ids))
    report.run('evidence-disjoint', lambda: check_disjoint(out, rows, ids, ready_json))
    report.run('admission-log', lambda: check_admission_log(server_log, rows,
                                                            [f for f in failures if f not in probe_failures],
                                                            frozenset(probe_keys), out=out))
    report.run('goal-async', lambda: check_goal_async(out, server_log, rows, policy, expect))
    running = expect.get('running')
    report.run('queue-wait-consistent', lambda: check_queue_waits(rows, timeline, queue_tolerance(running or n)))
    report.run('no-preemption', lambda: check_no_preemption(out))
    report.run('kv-pools', lambda: check_kv_pools(out, rows, expect))
    report.run('page-evidence', lambda: check_page_evidence(out, rows, expect))
    report.run('gdn-bytes', lambda: check_gdn_bytes(out, rows, expect))
    report.run('phase1-mode', lambda: check_phase1_mode(out, rows, expect))
    stops = [c['stop_reason'] for c in clients]
    if policy in SELECTIVE and rows:
        name = 'goal-rewriter-attempted' if policy in rewriter_policies(expect) else 'goal-rewriter-unused'
        report.run(name, lambda: check_goal_rewriter(policy, goals, expect, stops))
        report.run('goal-fallback', lambda: check_goal_fallback(out, rows, policy, expect))
        if policy == 'goal_query':
            report.run('goal-order', lambda: check_goal_order(out, rows, ids))
        if all(s == 'completed' for s in stops):
            # perf-int (scg2 r3): every session of a selective server selects at least once (pins.json selection_reach
            # proves it with the real tokenizer before the job; job 3338893's runs never selected).
            per_session = [sum(1 for x in turns if x['session'] == k and (x.get('selection') or {}).get('active'))
                           for k in range(n)]

            def exercised():
                need(active_turns >= 1, 'no turn reached active selection')
                need(all(per_session), 'sessions %s never reached active selection (active turns per session %s)'
                     % ([k for k, v in enumerate(per_session) if not v], per_session))
                return dict(active_turns=active_turns, per_session=per_session)
            report.run('query-selection-exercised', exercised)
    if running is not None:
        report.run('session-pool', lambda: check_session_pool(out, expect, rows, clients, ids, ready_json, timeline,
                                                              queue_tolerance(running), policy))
    report.run('cleanup', lambda: check_cleanup(out, ready_json, rows, ids, running is not None, probe_failures))
    session_turns = [c['completed_turns'] for c in clients]
    return dict(policy=policy, layout=layout['name'], status='PASS' if not report.failed else 'FAIL', failed=report.failed,
                findings=report.findings, checks=report.checks, turns=turns, sessions=n, session_ids=ids,
                session_turns=session_turns,
                planned_turns=clients[0]['planned_turns'] if clients else None, completed_turns=sum(session_turns),
                stop_reasons=stops, active_selection_turns=active_turns if policy in SELECTIVE else None,
                token_mode='exact' if tokens.exact else 'decode-only',
                # perf-ctx: each session's own planned turns (a long-context session ends at its long-reply turn) and
                # the long session's plan (growth listings, the long-reply turn).
                session_planned=[c.get('planned_turns') for c in clients],
                context_plan=(clients[0].get('plan') if clients and expect.get('context') else None))


# ---------------------------------------------------------------- perf-adm: the admission flags (adm-e)

ADMISSION_FIELDS = ('phase1_mode', 'phase1_park_ms', 'phase1_step_ns', 'phase1_step_rows')
PHASE1_MODES = ('mixed', 'probe_only', 'fallback')
# One GDN slot: 3 Mamba groups x 16 layers x 3,207,168 B (the carry; a hand-off or a crossing copies one slot).
GDN_BYTES = CARRY_H2D_BYTES


def admission_of(expect):
    """The run's admission record ({flag: True}), {} when every flag is off."""
    return dict(expect.get('admission') or {})


def gdn_handoff(expect):
    return admission_of(expect).get('gdn_device_handoff') is True


def overlap_permits(expect):
    """K of the run's (f) --admission-overlap (its pinned record's permits; 1 when off)."""
    return (admission_of(expect).get('admission_overlap') or {}).get('permits') or 1


def phase1_injector(expect):
    """Whether a server's phase 1 may run inside a step: --phase1-in-step with several sessions (a one-session server
    has no park table: every phase 1 is the separate probe-only pass)."""
    seqs = expect.get('max_num_seqs') or expect.get('running') or len(expect.get('tasks') or ())
    return admission_of(expect).get('phase1_in_step') is True and seqs > 1


def check_admission_flags(out, expect):
    """startup.json, READY and the worker's attach receipt name exactly the run's admission record (admission_flags:
    the engine and the worker verified it once at startup); with every flag off none of them has the key."""
    record = expect.get('admission')
    startup, ready, attached = load(out / 'startup.json'), load(out / 'ready.json'), load(out / 'worker-attached.json')
    seen = dict(startup=startup.get('admission'), ready=ready.get('admission'),
                worker=attached['receipt'].get('admission'))
    if record is None:
        present = [k for k, v in (('startup', startup), ('ready', ready), ('worker', attached['receipt'])) if 'admission' in v]
        need(not present, 'a flag-off server records admission flags in %s: %s' % (present, seen))
        return dict(admission=None)
    need(all(v == record for v in seen.values()), 'admission records %s, expected %s everywhere' % (seen, record))
    return dict(admission=record)


def gdn_crossings(start, end, mamba_tokens):
    """The Mamba-block crossings of a turn's phase 2 [start, end): the multiples of M strictly inside, and one more when
    phase 2 starts on a multiple (both allowed: no GPU turn has shown that edge yet)."""
    inside = max(0, (end - 1) // mamba_tokens - start // mamba_tokens) if end > start else 0
    edge = inside + (1 if 0 < start < end and start % mamba_tokens == 0 else 0)
    return sorted({inside, edge})


def check_gdn_bytes(out, rows, expect):
    """The GDN bytes of every turn (decoupled layouts; a legacy server is not checked): prepare H2D = the carry on a
    continuing turn, 0 at turn 0; flag off: restore H2D = restored Pages x P x 65,536 + GDN x (1 + k) (turn 0:
    GDN x k), k = the crossing copies, 0 <= k <= the Mamba-block boundaries inside phase 2 (one copy per engine step
    that crosses a block, perf-swe16) and no d2d field; --gdn-device-handoff: restore H2D = the restored Pages only and restore D2D =
    GDN x (1 + crossings) (turn 0: GDN x crossings), equal to the frames' own d2d counters (last after - first
    before). Returns the totals."""
    out = Path(out)
    layout = layout_of(expect)
    if not layout['decoupled']:
        return dict(layout='legacy', checked=False)
    handoff = gdn_handoff(expect)
    page = layout['page_tokens'] * FA_TOKEN_BYTES
    M = layout['mamba_tokens']
    totals = dict(turns=0, continuing=0, crossings=0, handoffs=0, restored_pages=0, prepare_h2d_bytes=0,
                  restore_h2d_bytes=0, d2d_bytes=0)
    for row in rows:
        native = load(native_path(out, row['request_id']))
        # perf-swe16: a real agent's session may rebuild (a format-error retry): the GDN carry and phase 1 follow the
        # session-local turn (stage_b.local_turn: 0 at a session start or rebuild), which equals the turn without one.
        stage = row['stage_b']
        t = stage['local_turn']
        where = 's%s turn %s' % (row.get('session_index'), t)
        restored = len(stage['restore'])
        # perf-tb16 (--release-idle-pages): the Pages readmitted for this request come back in its restore H2D (the
        # runtime's restore = every joined byte but phase 1's), one Page each
        restored += len(((row.get('idle_release') or {}).get('readmitted_pages')) or ())
        counts = gdn_crossings(stage['phase2_start'], phase2_end(stage), M)
        # perf-swe16 (dry run, job 3354132): one GDN copy per engine step that crosses a Mamba block, not per block
        # boundary; a phase 2 prefilled in one step crosses several blocks with a single copy. So any k crossing
        # copies with 0 <= k <= the boundaries inside phase 2 (turn 0 of a 3,850-token prompt measured 1 of 4).
        wanted = [GDN_BYTES * ((1 if t else 0) + k) for k in range(max(counts) + 1)]
        need(row['prepare']['h2d_bytes'] == (GDN_BYTES if t else 0), '%s: prepare H2D %d, expected the carry %d'
             % (where, row['prepare']['h2d_bytes'], GDN_BYTES if t else 0))
        restore = row['restore']
        if handoff:
            d2d = restore.get('d2d_bytes')
            need(restore['h2d_bytes'] == restored * page, '%s: restore H2D %d, expected the %d restored Pages only (%d)'
                 % (where, restore['h2d_bytes'], restored, restored * page))
            need(d2d in wanted, '%s: restore D2D %r, expected GDN x (%s + crossings %s) = %s'
                 % (where, d2d, 1 if t else 0, counts, wanted))
            frames = native['worker_frames']
            own = frames[-1]['http_measurement']['after'].get(D2D_KEY, 0) - \
                frames[0]['http_measurement']['before'].get(D2D_KEY, 0)
            need(own == d2d, '%s: the frames\' own D2D %d differs from the row\'s %d' % (where, own, d2d))
            crossings = d2d // GDN_BYTES - (1 if t else 0)
        else:
            need('d2d_bytes' not in restore, '%s: a flag-off row carries restore d2d_bytes' % where)
            extra = restore['h2d_bytes'] - restored * page
            need(extra in wanted, '%s: restore H2D %d = %d restored Pages + %d, expected GDN x (%s + crossings %s) = %s'
                 % (where, restore['h2d_bytes'], restored, extra, 1 if t else 0, counts, wanted))
            crossings = extra // GDN_BYTES - (1 if t else 0)
        totals['turns'] += 1
        totals['continuing'] += 1 if t else 0
        totals['handoffs'] += 1 if (t and handoff) else 0
        totals['crossings'] += crossings
        totals['restored_pages'] += restored
        totals['prepare_h2d_bytes'] += row['prepare']['h2d_bytes']
        totals['restore_h2d_bytes'] += restore['h2d_bytes']
        totals['d2d_bytes'] += restore.get('d2d_bytes') or 0
    return dict(totals, handoff=handoff, page_bytes=page, mamba_tokens=M, gdn_bytes=GDN_BYTES)


def check_phase1_mode(out, rows, expect):
    """How phase 1 ran (--phase1-in-step) as the native records and the frames show it: off: none of the in-step
    fields and no arm frame; on a one-session server: no arm frame, phase1_mode absent or probe_only, nothing parked
    (the separate pass, bit for bit); with several sessions: every continuing turn names its mode, a mixed one parked,
    ran in a step of 2..S rows for phase1_step_ns and has the arm frame (the phase-1 frame measured from it), the others
    neither. Returns the modes, parks and step times."""
    out = Path(out)
    flags = admission_of(expect)
    in_step, injector = flags.get('phase1_in_step') is True, phase1_injector(expect)
    seqs = expect.get('max_num_seqs') or expect.get('running') or len(expect['tasks'])
    modes, parked, steps, widths = {}, 0, [], []
    for row in rows:
        native = load(native_path(out, row['request_id']))
        t = row['stage_b']['local_turn']  # perf-swe32: the session-local turn (a rebuild is a local turn 0)
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        frames = native['worker_frames']
        ops = [f['operation'] for f in frames]
        fields = sorted(k for k in ADMISSION_FIELDS if k in native)
        mode = native.get('phase1_mode')
        if t:
            modes[str(mode)] = modes.get(str(mode), 0) + 1
        if not in_step or not injector or t == 0:
            need('phase1_arm' not in ops, '%s: a phase1_arm frame on a server without the step injector' % where)
        if not in_step:
            need(not fields, '%s: a flag-off native record carries %s' % (where, fields))
            continue
        if not injector or t == 0:
            need(set(fields) <= {'phase1_mode'} and mode in (None, 'probe_only') and (t or not fields),
                 '%s: %s server records %s (mode %r): its phase 1 is the separate probe-only pass'
                 % (where, 'a one-session' if not injector else 'turn 0 of an in-step', fields, mode))
            continue
        need(mode in PHASE1_MODES, '%s: phase1_mode %r, expected one of %s' % (where, mode, PHASE1_MODES))
        park = native.get('phase1_park_ms')
        parked += 'phase1_park_ms' in native
        if mode == 'mixed':
            step, width = native.get('phase1_step_ns'), native.get('phase1_step_rows')
            need(type(park) in (int, float) and math.isfinite(park) and park >= 0,
                 '%s: a mixed phase 1 without its parked time (%r)' % (where, park))
            need(type(step) is int and step > 0 and type(width) is int and 2 <= width <= seqs,
                 '%s: mixed step %r ns of %r rows, expected > 0 ns and 2..%d rows' % (where, step, width, seqs))
            need(ops[:5] == ['prepare', 'phase1_arm', 'phase1', 'phase1_drain', 'reserve'],
                 '%s: a mixed phase 1 with frames %s' % (where, ops[:6]))
            need(frames[2]['http_measurement']['before'] == frames[1]['http_measurement']['after'],
                 '%s: the in-step phase-1 frame is not measured from its arm' % where)
            steps.append(step)
            widths.append(width)
        else:
            need('phase1_arm' not in ops and 'phase1_step_ns' not in native and 'phase1_step_rows' not in native,
                 '%s: a %s phase 1 with an arm frame or step fields' % (where, mode))
            need(park is None or (type(park) in (int, float) and math.isfinite(park) and park >= 0),
                 '%s: phase1_park_ms %r' % (where, park))
    return dict(in_step=in_step, injector=injector, modes=dict(sorted(modes.items())), parked=parked,
                mixed=len(steps), step_ms_min=round(min(steps) / 1e6, 3) if steps else None,
                step_ms_max=round(max(steps) / 1e6, 3) if steps else None,
                step_rows_max=max(widths) if widths else None)


# ---------------------------------------------------------------- smoke-dp1: the decoupled KV layout

LAYOUT_KEY = 'remote_ra_kv_layout'
POOL_FIELDS = ('kv_layout', 'num_blocks_by_pool', 'free_blocks_by_pool')
DECOUPLED_CONFIG_KEYS = ('block_size', 'mamba_block_size', 'additional_config')


def additional_config(value):
    """--additional-config as the effective server config records it (a dict, or its JSON text)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def check_kv_layout(out, expect):
    """READY, the startup config and the worker attachment name the server's KV layout (interface I0).

    Legacy: none of kv_layout / num_blocks_by_pool / free_blocks_by_pool / kv_pools anywhere, the effective
    server config without block_size / mamba_block_size / additional_config, startup page_tokens 784 (the evidence
    of the shared pool, unchanged). Decoupled: the startup config's page_tokens and kv_layout {page_tokens P,
    mamba_tokens M, mamba_slots 2S + 1}, READY's kv_layout equal, both pools' sizes per DESIGN D4 (N_fa = (budget -
    Mamba bytes) // (P x 65,536), N_m = 1 + 3K), the readiness free blocks every block but each pool's null block,
    worker-attached.json's kv_pools the same, and the effective config's block_size P, mamba_block_size M and
    additional_config {"remote_ra_kv_layout": kv_layout}."""
    layout = layout_of(expect)
    ready, startup, attached = load(out / 'ready.json'), load(out / 'startup.json'), load(out / 'worker-attached.json')
    effective = ready.get('effective_server_config') or {}
    if not layout['decoupled']:
        present = [k for k in POOL_FIELDS if k in ready] + (['startup kv_layout'] if 'kv_layout' in startup else []) + \
            (['worker kv_pools'] if 'kv_pools' in attached else []) + [k for k in DECOUPLED_CONFIG_KEYS if k in effective]
        need(not present, 'legacy server records decoupled-layout evidence: %s' % present)
        need(startup.get('page_tokens') == PAGE, 'legacy startup page_tokens %r' % startup.get('page_tokens'))
        return dict(layout='legacy', page_tokens=PAGE)
    record, pools = layout['record'], list(layout['pools'])
    baseline = [n - 1 for n in pools]
    need(startup.get('page_tokens') == layout['page_tokens'] and startup.get('kv_layout') == record,
         'startup page_tokens %r / kv_layout %r, expected %d / %r'
         % (startup.get('page_tokens'), startup.get('kv_layout'), layout['page_tokens'], record))
    need(ready.get('kv_layout') == record, 'READY kv_layout %r, expected %r' % (ready.get('kv_layout'), record))
    need(ready.get('num_blocks_by_pool') == pools, 'pool sizes (FA, Mamba) %r, expected %r (DESIGN D4)'
         % (ready.get('num_blocks_by_pool'), pools))
    need(ready.get('free_blocks_by_pool') == baseline, 'readiness free blocks %r, expected %r (all but the null blocks)'
         % (ready.get('free_blocks_by_pool'), baseline))
    need(attached.get('kv_pools') == dict(kv_layout=record, num_blocks_by_pool=pools, free_blocks_by_pool=baseline),
         'worker-attached.json kv_pools %r differ from READY' % (attached.get('kv_pools'),))
    # A dict, not its JSON text: vLLM parsed --additional-config (the observed arguments equal the startup ones).
    need(effective.get('block_size') == layout['page_tokens'] and effective.get('mamba_block_size') == layout['mamba_tokens']
         and type(effective.get('additional_config')) is dict
         and effective['additional_config'] == {LAYOUT_KEY: record},
         'effective server config block_size %r / mamba_block_size %r / additional_config %r, expected %d / %d / %r'
         % (effective.get('block_size'), effective.get('mamba_block_size'), effective.get('additional_config'),
            layout['page_tokens'], layout['mamba_tokens'], {LAYOUT_KEY: record}))
    return dict(layout=layout['name'], page_tokens=layout['page_tokens'], mamba_tokens=layout['mamba_tokens'],
                mamba_slots=record['mamba_slots'], num_blocks_by_pool=pools, fa_tokens=pools[0] * layout['page_tokens'])


LEGACY_REPRESENTATION = 'pre_final_norm_last_token_l2'
HEX64_CHARS = frozenset('0123456789abcdef')
PAGE_RECORD_KEYS = {'namespace', 'binding', 'source', 'content'}


def float_row(value):
    return type(value) is list and bool(value) and all(type(x) is float and math.isfinite(x) for x in value)


def compact_vector(value):
    """compact_int8_v1 wire form: [[q_0..q_127] ints in [-127, 127], not all zero, finite positive float scale]."""
    return (type(value) is list and len(value) == 2 and type(value[0]) is list and len(value[0]) == 128
            and all(type(q) is int and -127 <= q <= 127 for q in value[0]) and any(value[0])
            and type(value[1]) is float and math.isfinite(value[1]) and value[1] > 0)


def check_page_evidence(out, rows, expect):
    """Page records and Page vectors as the worker sent them (native terminal files' worker frames).

    Legacy: every sealed Page record names its tokens by `token_sha256` (64 hex), the phase-1 receipt and the
    phase-2 publication carry `pre_final_norm_last_token_l2` float rows. Decoupled: the record names its tokens by
    `token_span` [pP, (p+1)P] instead (WP2: no per-Page hash at small P), and the Page vectors are
    `compact_int8_v1` ([[128 int8], scale], WP3: compacted on the device, interface I3) in the phase-1 receipt and the
    publication; the observation query stays a float row."""
    layout = layout_of(expect)
    P, decoupled = layout['page_tokens'], layout['decoupled']
    want = COMPACT_INT8_V1 if decoupled else LEGACY_REPRESENTATION
    vector_ok = compact_vector if decoupled else float_row
    pages = vectors = 0
    for row in rows:
        native = load(native_path(out, row['request_id']))
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        frames = native['worker_frames']
        for frame in frames:
            if frame['operation'] != 'phase1':
                continue
            need(frame.get('representation') == want, '%s: phase-1 representation %r, expected %r'
                 % (where, frame.get('representation'), want))
            need(float_row(frame.get('observation_vector')), '%s: phase-1 observation query is not a float row' % where)
            for page, value in (frame.get('page_vectors') or {}).items():
                need(vector_ok(value), '%s: phase-1 Page %s vector is not %s' % (where, page, want))
                vectors += 1
        done = frames[-1]
        for page, record in (done.get('new_pages') or {}).items():
            p = int(page)
            if decoupled:
                need(type(record) is dict and set(record) == PAGE_RECORD_KEYS | {'token_span'}
                     and record['token_span'] == [p * P, (p + 1) * P],
                     '%s: sealed Page %s record %r, expected token_span [%d, %d]' % (where, page, record, p * P, (p + 1) * P))
            else:
                need(type(record) is dict and set(record) == PAGE_RECORD_KEYS | {'token_sha256'}
                     and type(record['token_sha256']) is str and len(record['token_sha256']) == 64
                     and set(record['token_sha256']) <= HEX64_CHARS, '%s: sealed Page %s record %r' % (where, page, record))
            pages += 1
        body = (done.get('publication') or {}).get('body')
        need(isinstance(body, dict) and body.get('representation') == want, '%s: publication representation %r, expected %r'
             % (where, (body or {}).get('representation'), want))
        for page, value in (body.get('page_vectors') or {}).items():
            need(vector_ok(value), '%s: published Page %s vector is not %s' % (where, page, want))
            vectors += 1
    return dict(representation=want, page_records=pages, page_vectors=vectors)


def check_kv_pools(out, rows, expect):
    """Both pools per request and after drain (decoupled; a legacy server records none of it).

    Every runtime row carries its native terminal file's kv_pools (the free blocks after that request's drain);
    each sample lies within its pool, and the Mamba pool never holds more than two GDN slots (6 blocks) per session
    (the two-pool model: N_m - 1 = 3(2S + 1) blocks). After every session closed, engine-cleanup.json shows both
    pools back at the readiness baseline (no block leaked in either pool)."""
    layout = layout_of(expect)
    cleanup = load(out / 'engine-cleanup.json')
    natives = {row['request_id']: load(native_path(out, row['request_id'])) for row in rows}
    if not layout['decoupled']:
        present = [row['request_id'] for row in rows if 'kv_pools' in row or 'kv_pools' in natives[row['request_id']]]
        need(not present and 'kv_pools' not in cleanup, 'legacy server records pool samples (%d requests%s)'
             % (len(present), ', engine cleanup' if 'kv_pools' in cleanup else ''))
        return dict(layout='legacy')
    pools = list(layout['pools'])
    baseline = [n - 1 for n in pools]
    sessions = expect.get('running') or len(expect['tasks'])
    slots = 3 if gdn_handoff(expect) else 2  # perf-adm: the held probe slot until the first phase-2 step copied it
    samples = []
    for row in rows:
        value = row.get('kv_pools')
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        need(isinstance(value, dict) and set(value) == {'free_blocks_by_pool'}
             and value == natives[row['request_id']].get('kv_pools'),
             '%s: runtime kv_pools %r differ from the native terminal file' % (where, value))
        free = value['free_blocks_by_pool']
        need(type(free) is list and len(free) == 2 and all(type(f) is int and 0 <= f <= b for f, b in zip(free, baseline)),
             '%s: pool sample %r outside the pools %r' % (where, free, baseline))
        need(baseline[1] - free[1] <= slots * GDN_SLOT_BLOCKS * sessions,
             '%s: the Mamba pool holds %d blocks, more than %d GDN slots for each of %d sessions'
             % (where, baseline[1] - free[1], slots, sessions))
        samples.append(free)
    need(cleanup.get('kv_pools') == dict(num_blocks_by_pool=pools, free_blocks_by_pool=baseline),
         'pools not back at the readiness baseline after drain: engine cleanup %r, baseline %r'
         % (cleanup.get('kv_pools'), dict(num_blocks_by_pool=pools, free_blocks_by_pool=baseline)))
    return dict(samples=len(samples), baseline=baseline,
                fa_used_max=max((baseline[0] - f[0] for f in samples), default=0),
                mamba_used_max=max((baseline[1] - f[1] for f in samples), default=0))


def check_startup_log(server_log, expect):
    """vLLM's hybrid alignment as the server log shows it (platforms/interface.py).

    Legacy: the attention block raised to 784 tokens and the Mamba page padded, no decoupled line. Decoupled (runs
    B and C; WP1's open GPU-only points): exactly one line "Decoupled KV layout: attention block P tokens, Mamba
    block M tokens, K Mamba slots, unpadded Mamba page." with this layout's values, i.e. nothing upstream
    (models/config.py, arg_utils.py) replaced the user-set --block-size / --mamba-block-size before the check, and
    neither the 784-token alignment nor a Mamba padding line (no re-pad in align mode; kv-layout checks the pool
    sizes of the unpadded page)."""
    text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    align, padding = ALIGN_LOG in text, PADDING_LOG in text
    lines = [line for line in text.splitlines() if DECOUPLED_LOG_PREFIX in line]
    layout = layout_of(expect)
    if layout['decoupled']:
        record = layout['record']
        wanted = DECOUPLED_LOG % (record['page_tokens'], record['mamba_tokens'], record['mamba_slots'])
        # The runtime logs this line on every call of the platform hook (unlike the legacy lines, logged only when
        # they change something), so several identical lines are fine; every one must carry this layout's values.
        need(lines and all(line.rstrip().endswith(wanted) for line in lines),
             'decoupled server log has %d layout lines %s, expected one or more, each ending %r'
             % (len(lines), [line[-160:] for line in lines[:3]], wanted))
        need(not align and not padding, 'decoupled server log shows the shared-pool alignment (%s%s)'
             % ('784-token block ' if align else '', 'Mamba padding' if padding else ''))
        return dict(decoupled_logged=True, mamba_block_size_kept=True, mamba_unpadded=True, alignment_logged=False,
                    padding_logged=False)
    need(align and padding, 'legacy server log lacks the 784-token alignment or the Mamba padding line')
    need(not lines, 'legacy server log has a decoupled layout line')
    return dict(decoupled_logged=False, alignment_logged=align, padding_logged=padding)


# ---------------------------------------------------------------- perf-tb16's Gemma servers: the window layout
# Gemma 4 31B IT (runtime gemma-tb: kv_layout.WindowLayout, window_capacity, idle_release, evidence.py): two pools,
# pool 0 = global Pages of P tokens of the 10 global layers (P x 81,920 B), pool 1 = the sliding windows of the 5 vLLM
# sliding groups (one window slot per open session plus one forward's transient); no Mamba / GDN state, no carry.
# --release-idle-pages: idle sessions' Pages are released under pressure (worker `evict`, capacity.jsonl) and
# readmitted before their next phase 1 (a `readmit` frame of that request, counted in its restore H2D).
WINDOW_LOG = ('Decoupled KV layout: full-attention and sliding-window blocks of %d tokens, window %d tokens, %d window '
              'slots, %d-token forwards.')
WINDOW_POOLS_LOG = re.compile(r'Decoupled KV cache pools: (\d+) full-attention blocks, (\d+) sliding-window blocks')
WINDOW_FORBIDDEN_CONFIG = ('mamba_block_size', 'mamba_cache_mode', 'gdn_prefill_backend')
CAPACITY_EVENTS = 'capacity.jsonl'
CAPACITY_SCHEMA = 'remote-ra-capacity-event-v1'
IDLE_RELEASE_KEY = 'idle_release'
IDLE_RELEASE_TOTALS = ('evictions', 'evicted_pages', 'readmissions', 'readmitted_pages')


def check_window_layout(out, expect):
    """READY, the startup config and the worker attachment name the window layout: startup page_tokens P and kv_layout
    {page_tokens, state sliding, window_tokens, window_slots K, forward_tokens}, READY's kv_layout equal, the pools
    [global Pages, sliding block ids] as planned (controller.gemma_pools) with every block but the null blocks free,
    the worker's kv_pools and kv_layout the same, and the effective config's block_size P, additional_config
    {"remote_ra_kv_layout": kv_layout}, the attention backend, no Mamba / GDN key."""
    layout = layout_of(expect)
    out = Path(out)
    ready, startup, attached = load(out / 'ready.json'), load(out / 'startup.json'), load(out / 'worker-attached.json')
    effective = ready.get('effective_server_config') or {}
    record, pools = layout['record'], list(layout['pools'])
    baseline = [n - 1 for n in pools]
    need(startup.get('page_tokens') == layout['page_tokens'] and startup.get('kv_layout') == record,
         'startup page_tokens %r / kv_layout %r, expected %d / %r' % (startup.get('page_tokens'),
                                                                      startup.get('kv_layout'), layout['page_tokens'],
                                                                      record))
    need(ready.get('kv_layout') == record, 'READY kv_layout %r, expected %r' % (ready.get('kv_layout'), record))
    need(ready.get('num_blocks_by_pool') == pools and ready.get('free_blocks_by_pool') == baseline,
         'pool sizes (global Pages, sliding blocks) %r / free %r, expected %r / %r'
         % (ready.get('num_blocks_by_pool'), ready.get('free_blocks_by_pool'), pools, baseline))
    need(attached.get('kv_pools') == dict(kv_layout=record, num_blocks_by_pool=pools, free_blocks_by_pool=baseline),
         'worker-attached.json kv_pools %r differ from READY' % (attached.get('kv_pools'),))
    receipt = attached.get('receipt') or {}
    need(receipt.get('kv_layout') in (None, {LAYOUT_KEY: record}), 'worker kv_layout %r' % (receipt.get('kv_layout'),))
    need(effective.get('block_size') == layout['page_tokens'] and type(effective.get('additional_config')) is dict
         and effective['additional_config'] == {LAYOUT_KEY: record}
         and not [k for k in WINDOW_FORBIDDEN_CONFIG if k in effective]
         and effective.get('attention_backend') == expect.get('attention_backend'),
         'effective server config block_size %r / additional_config %r / %s / attention_backend %r, expected %d / %r / '
         'no Mamba keys / %r' % (effective.get('block_size'), effective.get('additional_config'),
                                 [k for k in WINDOW_FORBIDDEN_CONFIG if k in effective],
                                 effective.get('attention_backend'), layout['page_tokens'], {LAYOUT_KEY: record},
                                 expect.get('attention_backend')))
    return dict(layout=layout['name'], page_tokens=layout['page_tokens'], window_slots=record['window_slots'],
                num_blocks_by_pool=pools, global_tokens=pools[0] * layout['page_tokens'])


def check_window_startup_log(server_log, expect):
    """The window layout as the server log shows it (the overlay's platform hook, kv_cache_utils): every
    "Decoupled KV layout:" line names these blocks, window, slots and forwards; the pools line names the planned pools;
    neither the 784-token alignment nor a Mamba padding line."""
    text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    layout = layout_of(expect)
    record = layout['record']
    wanted = WINDOW_LOG % (record['page_tokens'], record['window_tokens'], record['window_slots'],
                           record['forward_tokens'])
    lines = [line for line in text.splitlines() if DECOUPLED_LOG_PREFIX in line]
    need(lines and all(line.rstrip().endswith(wanted) for line in lines),
         'window server log has %d layout lines %s, expected one or more, each ending %r'
         % (len(lines), [line[-160:] for line in lines[:3]], wanted))
    pools = sorted({tuple(int(v) for v in m) for m in WINDOW_POOLS_LOG.findall(text)})
    need(pools == [tuple(layout['pools'])], 'window server log pools %s, expected %s' % (pools, layout['pools']))
    need(ALIGN_LOG not in text and PADDING_LOG not in text, 'window server log shows the shared-pool alignment')
    return dict(layout_lines=len(lines), pools=list(pools[0]), alignment_logged=False, padding_logged=False)


def check_window_pools(out, rows, expect):
    """Both pools per request and after drain (the window layout): every runtime row carries its native file's
    kv_pools; each sample lies within its pools (the sliding pool holds at most its K window slots and one forward's
    transient: its whole size); after every session closed engine-cleanup.json shows both pools at the readiness
    baseline (released and readmitted Pages included)."""
    out = Path(out)
    layout = layout_of(expect)
    cleanup = load(out / 'engine-cleanup.json')
    pools = list(layout['pools'])
    baseline = [n - 1 for n in pools]
    samples = []
    for row in rows:
        native = load(native_path(out, row['request_id']))
        value = row.get('kv_pools')
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        need(isinstance(value, dict) and set(value) == {'free_blocks_by_pool'} and value == native.get('kv_pools'),
             '%s: runtime kv_pools %r differ from the native terminal file' % (where, value))
        free = value['free_blocks_by_pool']
        need(type(free) is list and len(free) == 2 and all(type(f) is int and 0 <= f <= b for f, b in zip(free, baseline)),
             '%s: pool sample %r outside the pools %r' % (where, free, baseline))
        samples.append(free)
    need(cleanup.get('kv_pools') == dict(num_blocks_by_pool=pools, free_blocks_by_pool=baseline),
         'pools not back at the readiness baseline after drain: engine cleanup %r, baseline %r'
         % (cleanup.get('kv_pools'), dict(num_blocks_by_pool=pools, free_blocks_by_pool=baseline)))
    return dict(samples=len(samples), baseline=baseline,
                global_used_max=max((baseline[0] - f[0] for f in samples), default=0),
                sliding_used_max=max((baseline[1] - f[1] for f in samples), default=0))


def idle_release_of(row):
    return row.get(IDLE_RELEASE_KEY) or dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[])


def check_page_bytes(out, rows, expect):
    """The H2D bytes of every turn of a window-layout server (check_gdn_bytes' place: no GDN state, no carry): prepare
    H2D 0 on every turn; restore H2D = (the restored Pages + the Pages readmitted for this request) x P x 81,920; no
    D2D. Returns the totals."""
    out = Path(out)
    layout = layout_of(expect)
    page = layout['page_bytes']
    totals = dict(turns=0, restored_pages=0, readmitted_pages=0, restore_h2d_bytes=0)
    for row in rows:
        where = 's%s turn %s' % (row.get('session_index'), row.get('turn'))
        restored = len(row['stage_b']['restore'])
        readmitted = len(idle_release_of(row)['readmitted_pages'])
        need(row['prepare']['h2d_bytes'] == 0, '%s: prepare H2D %d on a window layout (no carry)'
             % (where, row['prepare']['h2d_bytes']))
        need('d2d_bytes' not in row['restore'], '%s: restore d2d_bytes on a window layout' % where)
        need(row['restore']['h2d_bytes'] == (restored + readmitted) * page,
             '%s: restore H2D %d, expected (%d restored + %d readmitted Pages) x %d'
             % (where, row['restore']['h2d_bytes'], restored, readmitted, page))
        totals['turns'] += 1
        totals['restored_pages'] += restored
        totals['readmitted_pages'] += readmitted
        totals['restore_h2d_bytes'] += row['restore']['h2d_bytes']
    return dict(totals, page_bytes=page)


def window_frames(native, row):
    """A window-layout request's worker frames without its `readmit` frame, which --release-idle-pages adds right after
    `prepare` on a continuing turn of a released session: its receipt names exactly the row's readmitted Pages,
    drained, under the request's binding. Returns the other frames (frame_ops' order)."""
    frames = native['worker_frames']
    readmits = [i for i, f in enumerate(frames) if f['operation'] == 'readmit']
    pages = idle_release_of(row)['readmitted_pages']
    need(len(readmits) == (1 if pages else 0), 'readmit frames at %s for readmitted Pages %r' % (readmits, pages))
    if readmits:
        frame = frames[readmits[0]]
        need(readmits == [1] and native['local_turn'] > 0 and frame.get('readmitted') == pages
             and frame.get('drained') is True and frame.get('binding') == frames[0].get('binding'),
             'readmit frame %r at %s (local turn %s, readmitted Pages %r)' % (
                 {k: frame.get(k) for k in ('operation', 'readmitted', 'drained')}, readmits, native['local_turn'],
                 pages))
    return [f for f in frames if f['operation'] != 'readmit']


def check_idle_release(out, rows, expect):
    """--release-idle-pages as the server's evidence shows it. On: startup.json, READY and the worker receipt pin
    release_idle_pages true; every row (and its native file) carries its idle_release record {evicted_sessions,
    evicted_pages, readmitted_pages}; capacity.jsonl rows are evict / readmit events of one FA group (blocks = Pages)
    that leave the sliding pool unchanged; a readmit names a completed request with exactly those Pages;
    engine-cleanup.json's idle_release totals are the events' sums. Off: none of it exists."""
    out = Path(out)
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    receipt = load(out / 'worker-attached.json').get('receipt') or {}
    cleanup = load(out / 'engine-cleanup.json')
    path = out / CAPACITY_EVENTS
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] \
        if path.is_file() else []
    if not expect.get('release_idle_pages'):
        present = [n for n, v in (('startup', startup), ('ready', ready), ('worker', receipt)) if 'release_idle_pages' in v]
        need(not present and not events and IDLE_RELEASE_KEY not in cleanup
             and not any(IDLE_RELEASE_KEY in r for r in rows),
             'a server without --release-idle-pages records it (%s, %d capacity events)' % (present, len(events)))
        return dict(release_idle_pages=False)
    seen = dict(startup=startup.get('release_idle_pages'), ready=ready.get('release_idle_pages'),
                worker=receipt.get('release_idle_pages'))
    need(all(v is True for v in seen.values()), 'release_idle_pages records %s, expected true everywhere' % seen)
    by_request = {}
    for row in rows:
        native = load(native_path(out, row['request_id']))
        value = row.get(IDLE_RELEASE_KEY)
        need(isinstance(value, dict) and set(value) == {'evicted_sessions', 'evicted_pages', 'readmitted_pages'}
             and value == native.get(IDLE_RELEASE_KEY) and type(value['evicted_pages']) is int
             and all(type(s) is str for s in value['evicted_sessions'])
             and all(type(p) is int and p >= 0 for p in value['readmitted_pages']),
             'row %s: idle_release %r (native %r)' % (row.get('request_id'), value, native.get(IDLE_RELEASE_KEY)))
        by_request[row['request_id']] = value
    totals = dict.fromkeys(IDLE_RELEASE_TOTALS, 0)
    for e in events:
        need(e.get('schema') == CAPACITY_SCHEMA and e.get('event') in ('evict', 'readmit')
             and type(e.get('pages')) is int and e['pages'] > 0 and e.get('blocks') == e['pages']
             and type(e.get('monotonic_ns')) is int
             and isinstance(e.get('free_blocks_by_pool_before'), list) and isinstance(e.get('free_blocks_by_pool'), list)
             and e['free_blocks_by_pool_before'][1:] == e['free_blocks_by_pool'][1:],
             'malformed capacity.jsonl row %r' % ({k: e.get(k) for k in ('event', 'pages', 'blocks')},))
        if e['event'] == 'evict':
            totals['evictions'] += 1
            totals['evicted_pages'] += e['pages']
        else:
            totals['readmissions'] += 1
            totals['readmitted_pages'] += e['pages']
            row = by_request.get(e.get('request_id'))
            need(row is None or len(row['readmitted_pages']) == e['pages'],
                 'readmit of %s: %d Pages, the row readmitted %s' % (e.get('request_id'), e['pages'], row))
    need([e['monotonic_ns'] for e in events] == sorted(e['monotonic_ns'] for e in events),
         'capacity.jsonl rows are not in time order')
    need(cleanup.get(IDLE_RELEASE_KEY) == totals, 'engine cleanup idle_release %r, the events sum to %r'
         % (cleanup.get(IDLE_RELEASE_KEY), totals))
    rows_readmitted = sum(len(v['readmitted_pages']) for v in by_request.values())
    need(rows_readmitted <= totals['readmitted_pages'], 'rows readmitted %d Pages, the events %d'
         % (rows_readmitted, totals['readmitted_pages']))
    return dict(totals, rows_readmitted_pages=rows_readmitted,
                rows_with_evictions=sum(1 for v in by_request.values() if v['evicted_sessions']))


# ---------------------------------------------------------------- smoke-dp1: checks over a plan's runs

OVERHEAD_FINDING = 'OVERHEAD_ABOVE_BUDGET'
ADMISSION_FINDING = 'ADMISSION_SLOWER'
RAN = ('PASS', 'PASS_TRUNCATED')


def server_turns(out):
    """Per (session, turn) of one server: prompt token ids, reply content, completion tokens, H2D split, admission."""
    out = Path(out)
    rows = {r['response_id']: r for r in measure.runtime_rows(out)}
    turns = {}
    k = 0
    while (out / ('s%d' % k) / 'session.json').exists():
        client = load(out / ('s%d' % k) / 'session.json')
        for t in range(client['completed_turns']):
            response = load(out / ('s%d' % k) / ('http-response-%d.json' % t))
            row = rows.get(response.get('id'))
            need(row is not None, 's%d turn %d of %s has no runtime row' % (k, t, out.name))
            native = load(native_path(out, row['request_id']))
            turns[(k, t)] = dict(prompt=row['prompt_token_ids'], content=response['choices'][0]['message'].get('content'),
                                 completion=(response.get('usage') or {}).get('completion_tokens'),
                                 prepare=row['prepare']['h2d_bytes'], restore=row['restore']['h2d_bytes'],
                                 admission_ms=native.get('admission_engine_ms'))
        k += 1
    return turns


def first_divergence(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def check_exact(reference, other):
    """`other`'s tokens equal the reference run's (DESIGN section 8): every turn's prompt token ids (which hold every
    earlier reply), reply text and completion count, on every turn both servers completed. A divergence names its
    first session, turn and token position."""
    a, b = server_turns(reference), server_turns(other)
    common = sorted(set(a) & set(b))
    need(common, 'no turn completed by both %s and %s' % (Path(reference).name, Path(other).name))
    for key in common:
        x, y = a[key], b[key]
        position = first_divergence(x['prompt'], y['prompt'])
        if position is not None:
            need(False, 's%d turn %d: prompt tokens diverge at position %d (%s %d tokens, %s %d)'
                 % (key[0], key[1], position, Path(reference).name, len(x['prompt']), Path(other).name, len(y['prompt'])))
        need(x['content'] == y['content'] and x['completion'] == y['completion'],
             's%d turn %d: reply differs (%r tokens vs %r)' % (key[0], key[1], x['completion'], y['completion']))
    return dict(turns=len(common), reference_only=sorted(set(a) - set(b)), other_only=sorted(set(b) - set(a)),
                prompt_tokens_last=len(a[common[-1]]['prompt']))


def check_h2d(reference, other):
    """Phase-1 (carry restore) and other H2D bytes equal the reference run's on every common turn: the GDN carry
    (153,944,064 B per continuing turn) and the Mamba-block crossings (relocations) are the same at M = 784."""
    a, b = server_turns(reference), server_turns(other)
    common = sorted(set(a) & set(b))
    for key in common:
        need((a[key]['prepare'], a[key]['restore']) == (b[key]['prepare'], b[key]['restore']),
             's%d turn %d: H2D (prepare, restore) %s, reference %s' % (key[0], key[1], (b[key]['prepare'], b[key]['restore']),
                                                                    (a[key]['prepare'], a[key]['restore'])))
        need(key[1] == 0 or b[key]['prepare'] == CARRY_H2D_BYTES, 's%d turn %d: prepare H2D %d, expected the carry %d'
             % (key[0], key[1], b[key]['prepare'], CARRY_H2D_BYTES))
    return dict(turns=len(common), restore_bytes=sum(b[k]['restore'] for k in common))


def steady_itl(summary):
    """{b: (steady ITL ms, steady seconds)} of one run's batch profile."""
    batch = ((summary.get('measurements') or {}).get('batch') or {}).get('batches') or ()
    return {row['batch']: (row['steady_itl_ms'], row['steady_s']) for row in batch if row.get('steady_itl_ms') is not None}


def check_overhead(by_label, reference, budget_ms):
    """Per-step host overhead against the reference run (R5): per batch size b with steady intervals in both runs,
    the steady ITL difference and that difference per request (/ b). More than `budget_ms` per request is the
    finding OVERHEAD_ABOVE_BUDGET (a measured outcome, never a harness failure)."""
    base = steady_itl(by_label[reference])
    if not base:  # perf-ms16: reported, never a failure
        return dict(budget_ms_per_request=budget_ms, table=[], compared=0,
                    note='reference run %s has no steady decode interval' % reference)
    table, over = [], []
    for label, summary in by_label.items():
        if label == reference:
            continue
        for b, (itl, steady) in sorted(steady_itl(summary).items()):
            if b not in base:
                continue
            delta = round(itl - base[b][0], 3)
            row = dict(label=label, batch=b, steady_itl_ms=itl, reference_itl_ms=base[b][0], delta_ms=delta,
                       delta_per_request_ms=round(delta / b, 3), steady_s=steady, reference_steady_s=base[b][1])
            table.append(row)
            if row['delta_per_request_ms'] > budget_ms:
                over.append(row)
    if not table:  # perf-ms16: a measurement, never a failure (the 16-session runs may pass b <= 4 only in passing)
        return dict(budget_ms_per_request=budget_ms, table=[], compared=0,
                    note='no batch size with steady intervals in both the reference and another run')
    if over:
        raise Finding(OVERHEAD_FINDING, dict(budget_ms_per_request=budget_ms, table=table, over=over))
    return dict(budget_ms_per_request=budget_ms, table=table)


def check_admission(by_label, reference, labels, limit):
    """Admission (phase 1, selection and reserve on the engine thread) of the other runs against the reference: the
    mean admission_engine_ms of the continuing turns; slower by more than `limit` is the finding ADMISSION_SLOWER."""
    def mean_ms(summary):
        values = [t['admission_engine_ms'] for t in summary.get('turns') or () if t.get('turn') and
                  type(t.get('admission_engine_ms')) in (int, float)]
        return sum(values) / len(values) if values else None
    base = mean_ms(by_label[reference])
    need(base, 'reference run %s has no continuing turn with an admission time' % reference)
    table = []
    for label in labels:
        value = mean_ms(by_label[label])
        if value is not None:
            table.append(dict(label=label, admission_ms_mean=round(value, 3), reference_ms_mean=round(base, 3),
                              ratio=round(value / base, 4)))
    slower = [row for row in table if row['ratio'] > 1 + limit]
    if slower:
        raise Finding(ADMISSION_FINDING, dict(limit=limit, table=table, slower=slower))
    return dict(limit=limit, table=table)


def check_selection_p16(root, by_label):
    """The selective runs at P = 16: every query or random_fill run reached active selection with the PoC numbers (each
    turn's selection check already recomputed capacity = floor(0.4 x 16n) // 16, mandatory = ceil(initial / 16), at
    most 64 recent Pages, and random_fill's seeded draw), random50 kept floor(n / 2) Pages, and every selective run
    restored Pages on some turn (restore H2D beyond the GDN relocations)."""
    detail = {}
    for label, summary in by_label.items():
        policy = summary.get('policy')
        turns = summary.get('turns') or ()
        if policy in SELECTIVE:
            active = [t for t in turns if (t.get('selection') or {}).get('active')]
            need(active, '%s never reached active selection' % label)
            need(all((t['selection'].get('recent') or 0) <= 64 for t in active), '%s: more than 64 recent Pages' % label)
            detail[label] = dict(active_turns=len(active), capacity=[t['selection'].get('capacity') for t in active])
        if policy in SELECTIVE + ('random50',):
            restored = [t for t in turns if (t.get('stage_b') or {}).get('restore')]
            need(restored, '%s restored no Page on any turn' % label)
            detail.setdefault(label, {})['restore_turns'] = len(restored)
    return detail


BATCH_FINDING = 'NOT_BATCHED_16'  # at 16 sessions; NOT_BATCHED_<S> at another largest session count (batch_finding)


def batch_finding(sessions):
    return 'NOT_BATCHED_%d' % sessions
RUNNING_LOG = re.compile(r'Engine \d+: .*Running: (\d+) reqs, Waiting: (\d+) reqs')


def running_log(server_log):
    """vLLM's periodic stats lines (loggers.py, every 10 s) of one server log: {running count: lines}, max running."""
    counts = {}
    try:
        text = Path(server_log).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return dict(lines=0, counts={}, max_running=None)
    for found in RUNNING_LOG.finditer(text):
        counts[int(found.group(1))] = counts.get(int(found.group(1)), 0) + 1
    return dict(lines=sum(counts.values()), counts={str(k): v for k, v in sorted(counts.items())},
                max_running=max(counts) if counts else None)


def check_batched(by_label, sessions, logs=None):
    """The 16-session runs decode their sessions in one batch: on every run with the segment's largest session count
    (`sessions`, 16), the 1-s /metrics samples (vllm:num_requests_running) reach that many running requests. vLLM's own
    stats lines ("Running: N reqs", every 10 s; `logs` {label: server log}) are counted per run as a second view (a
    batch shorter than 10 s may fall between two lines). A run that never reached it (the sessions' turns did not
    overlap: a workload timing, not a runtime fault) is the finding NOT_BATCHED_16, a partial pass: the 16-session
    batch was not exercised. Smaller runs (R4) are reported, not required."""
    detail, low = {}, {}
    for label, summary in by_label.items():
        peak = ((summary.get('measurements') or {}).get('batch') or {}).get('running_peak')
        count = summary.get('running') or summary.get('sessions')  # a session pool batches its K running sessions
        seen = running_log((logs or {}).get(label)) if logs and label in logs else None
        detail[label] = dict(sessions=count, running_peak=peak, log=seen,
                             required=count == sessions)
        if count == sessions and not (type(peak) in (int, float) and peak >= sessions):
            low[label] = peak
    if not any(v['required'] for v in detail.values()):  # every 16-session run skipped: the batch was not exercised
        raise Finding(batch_finding(sessions), dict(sessions=sessions, runs=detail, below={}, ran=False))
    if low:
        raise Finding(batch_finding(sessions), dict(sessions=sessions, runs=detail, below=low))
    return dict(sessions=sessions, runs=detail)


CROSS_FINDING = 'BATCH_SHAPE_DIVERGENCE'


def check_cross_run(reference, other, sessions):
    """Sessions 0..sessions-1 of `other` (S16-full) against the same sessions of `reference` (R4): the same tasks and
    observations, so every turn's prompt token ids, reply text and completion count are equal until the first reply
    that differs. A reply that differs is the finding BATCH_SHAPE_DIVERGENCE (bf16 kernels give batch-shape-dependent
    numerics, so greedy decoding at batch 16 may pick another token than at batch 4; the session's later turns then
    differ by construction and are not compared); a prompt that differs although every earlier reply was equal (the
    render, routing or observation differs) is a failure, as is a session either run did not complete at turn 0."""
    a, b = server_turns(reference), server_turns(other)
    rows, diverged = [], []
    for k in range(sessions):
        common = sorted(t for (s, t) in set(a) & set(b) if s == k)
        need(0 in common, 's%d: turn 0 not completed by both %s and %s' % (k, Path(reference).name, Path(other).name))
        row = dict(session=k, turns_compared=0, identical_turns=0, first_divergence=None)
        for t in common:
            x, y = a[(k, t)], b[(k, t)]
            row['turns_compared'] += 1
            position = first_divergence(x['prompt'], y['prompt'])
            if position is not None:
                need(False, 's%d turn %d: prompt tokens differ at position %d after identical replies (%s %d tokens, '
                     '%s %d)' % (k, t, position, Path(reference).name, len(x['prompt']), Path(other).name,
                                 len(y['prompt'])))
            if x['content'] != y['content'] or x['completion'] != y['completion']:
                prefix = next((i for i, (p, q) in enumerate(zip(x['content'] or '', y['content'] or '')) if p != q),
                              min(len(x['content'] or ''), len(y['content'] or '')))
                row['first_divergence'] = dict(turn=t, completion_tokens=[x['completion'], y['completion']],
                                               first_differing_char=prefix)
                diverged.append(k)
                break
            row['identical_turns'] += 1
        rows.append(row)
    detail = dict(reference=Path(reference).name, other=Path(other).name, sessions=sessions, rows=rows,
                  identical_sessions=sessions - len(diverged),
                  identical_turns=sum(r['identical_turns'] for r in rows))
    if diverged:
        raise Finding(CROSS_FINDING, dict(detail, diverged_sessions=diverged))
    return detail


RESIDUAL_FINDING = 'RESIDUAL_SLOPE_HIGH'


def admission_phase1_ms(row):
    """The phase-1 ms inside a measurement row's engine-thread admission: phase1_ms, or 0 when the admission parked
    for its phase 1 (perf-adm --phase1-in-step: the probe ran in a step, or in a separate pass while parked; either way
    admission_engine_ms leaves the parked time out)."""
    return 0.0 if row.get('phase1_parked') else row.get('phase1_ms')


def residual_points(summary):
    """(candidate Pages, admission residual ms) of one run's valid continuing turns: the engine-thread admission
    (admission_engine_ms) minus its phase-1 forward (phase1_ms; none when it parked for an in-step phase 1),
    adm-vec's residual."""
    points = []
    for row in ((summary.get('measurements') or {}).get('rows') or ()):
        pages, stall, phase1 = (row.get('pages') or {}).get('candidates'), row.get('admission_engine_ms'), row.get('phase1_ms')
        if row.get('valid') and row.get('turn') and None not in (pages, stall, phase1):
            points.append((pages, round(stall - admission_phase1_ms(row), 3)))
    return points


def least_squares(points):
    """(slope, intercept, r2) of y = a x + b, or None with fewer than two distinct x."""
    n = len(points)
    if n < 2:
        return None
    mx, my = sum(p[0] for p in points) / n, sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx == 0:
        return None
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    syy = sum((p[1] - my) ** 2 for p in points)
    return slope, my - slope * mx, (sxy * sxy / (sxx * syy)) if syy else 1.0


def check_residual(by_label, limit_us, min_requests):
    """The admission residual per candidate Page (adm-vec: about 15-20 us/Page expected on the GH200 against the P = 16
    pilot's 31-62): per run with at least `min_requests` valid continuing turns, the least-squares slope of the
    residual (ms) over the candidate Pages, in us per Page, its intercept, r^2 and the Page range. A slope above
    `limit_us` is the finding RESIDUAL_SLOPE_HIGH (a measured outcome, never a failure)."""
    table, high = [], []
    for label, summary in by_label.items():
        points = residual_points(summary)
        fit = least_squares(points) if len(points) >= min_requests else None
        row = dict(label=label, policy=summary.get('policy'), sessions=summary.get('sessions'), requests=len(points),
                   pages_min=min((p[0] for p in points), default=None), pages_max=max((p[0] for p in points), default=None),
                   residual_ms_max=max((p[1] for p in points), default=None), slope_us_per_page=None, intercept_ms=None,
                   r2=None)
        if fit is not None:
            row.update(slope_us_per_page=round(fit[0] * 1000, 3), intercept_ms=round(fit[1], 3), r2=round(fit[2], 4))
            if row['slope_us_per_page'] > limit_us:
                high.append(row)
        table.append(row)
    if high:
        raise Finding(RESIDUAL_FINDING, dict(limit_us_per_page=limit_us, table=table, high=high))
    return dict(limit_us_per_page=limit_us, table=table,
                fitted=sum(1 for r in table if r['slope_us_per_page'] is not None))


def check_plan(root, options, pins, results):
    """One segment's checks over its runtime runs (job.plan_checks, postrun/verify_evidence.py): returns dict(status,
    checks, failed, findings, tables). The segment's `checks` (pins.json segments) choose them: `exact` (the listed runs
    token-identical to the reference, their H2D equal), `admission` (admission time against the reference: finding
    ADMISSION_SLOWER), `selection` (selection at P = 16 and restores), `batched` (16 running requests on every
    16-session run: finding NOT_BATCHED_16), `overhead` (per-step overhead against the reference at the batch sizes
    both reached: finding OVERHEAD_ABOVE_BUDGET), `cross-run` (sessions 0-3 of S16-full against R4: finding
    BATCH_SHAPE_DIVERGENCE) and `residual` (admission residual per Page: finding RESIDUAL_SLOPE_HIGH). A run that did
    not run (time budget, abort) leaves its comparisons out; its own status already decides the job."""
    root = Path(root)
    labels = [r['label'] for r in options['runs']]
    by_label = {r.get('label'): r for r in results if r.get('status') in RAN and r.get('label') in labels}
    reference, settings, wanted = options['reference'], pins['checks'], options['checks']
    report, tables = Report(), {}
    if 'exact' in wanted:
        for label in options['exact']:
            if reference in by_label and label in by_label:
                report.run('tokens-%s-vs-%s' % (label, reference),
                           lambda label=label: check_exact(root / reference, root / label))
                report.run('h2d-%s-vs-%s' % (label, reference), lambda label=label: check_h2d(root / reference, root / label))
    if 'admission' in wanted and reference in by_label:
        others = [label for label in labels if label != reference and label in by_label]
        tables['admission'] = report.run('admission-vs-' + reference, lambda: check_admission(
            by_label, reference, others, settings['admission_slowdown_max']))
    if 'selection' in wanted and any(s.get('policy') in SELECTIVE + ('random50',) for s in by_label.values()):
        report.run('selection-p16', lambda: check_selection_p16(root, by_label))
    if 'batched' in wanted:
        logs = {label: root.parent / ('server-' + label + '.log') for label in by_label}
        seqs = max((r.get('running') or r['sessions']) for r in options['runs'])
        tables['batched'] = report.run('batched-decode-%d' % seqs, lambda: check_batched(by_label, seqs, logs))
    if 'overhead' in wanted and reference in by_label and len(by_label) > 1:
        tables['overhead'] = report.run('step-overhead-vs-' + reference,
                                        lambda: check_overhead(by_label, reference, settings['overhead_ms_per_request']))
    cross = options.get('cross_run')
    if 'cross-run' in wanted and cross and cross['reference'] in by_label and cross['other'] in by_label:
        tables['cross_run'] = report.run('cross-run-%s-vs-%s' % (cross['other'], cross['reference']), lambda: check_cross_run(
            root / cross['reference'], root / cross['other'], cross['sessions']))
    if 'residual' in wanted:
        tables['residual'] = report.run('admission-residual', lambda: check_residual(
            by_label, settings['residual_us_per_page_max'], settings['residual_min_requests']))
    # perf-int: graph runs against the eager run (tokens and selections), selection under graph batching, job 3339019 as
    # the reference, the admission residual's components.
    for other, eager in options.get('compare') or ():
        if other in by_label and eager in by_label:
            tables['compare-%s-vs-%s' % (other, eager)] = report.run(
                'graph-tokens-%s-vs-%s' % (other, eager), lambda o=other, e=eager: compare_tokens(root / e, root / o))
            if by_label[other].get('policy') in QUERY:
                tables['selection-%s-vs-%s' % (other, eager)] = report.run(
                    'graph-selection-%s-vs-%s' % (other, eager),
                    lambda o=other, e=eager: compare_selection(root / e, root / o))
    if 'active-batched' in wanted:
        for label, summary in by_label.items():
            if summary.get('policy') in SELECTIVE and (summary.get('sessions') or 1) > 1:
                tables['active-batched-' + label] = report.run('active-batched-' + label,
                                                               lambda l=label: active_batching(root / l))
    reference = pins.get('reference_job')
    pairs = [(label, old) for label, old in sorted((options.get('reference_job') or {}).items())
             if (root / label / 's0' / 'session.json').exists()]
    if 'reference-job' in wanted and reference:
        for label, old in pairs:
            tables['reference-tokens-' + label] = report.run('reference-tokens-%s-vs-%s' % (label, old), lambda l=label, o=old:
                                                             check_reference_tokens(root / l, reference, l, o))
            if reference['runs'][old].get('policy') in SELECTIVE:
                tables['reference-selection-' + label] = report.run(
                    'reference-selection-%s-vs-%s' % (label, old),
                    lambda l=label, o=old: check_reference_selection(root / l, reference, l, o))
            if label in by_label:
                tables['reference-metrics-' + label] = report.run(
                    'reference-metrics-%s-vs-%s' % (label, old), lambda l=label, o=old: check_reference_metrics(
                        by_label[l], reference, l, o, settings['admission_slowdown_max']))
    if 'components' in wanted:
        tables['components'] = report.run('admission-components', lambda: check_components(
            by_label, reference, [(label, old) for label, old in pairs if label in by_label]))
    # perf-adm: the flag runs against the segment's flag-off reference.
    ref = options['reference']
    runs_by = {r['label']: r for r in options['runs']}
    for label in options.get('parity') or ():
        if ref in by_label and label in by_label:
            tables['parity-' + label] = report.run('parity-%s-vs-%s' % (label, ref),
                                                   lambda l=label: check_parity(root / ref, root / l))
    for label in options.get('repeat') or ():
        if ref in by_label and label in by_label:
            tables['repeat-' + label] = report.run('repeat-%s-vs-%s' % (label, ref),
                                                   lambda l=label: check_repeat(root / ref, root / l))
    for label in options.get('flag_compare') or ():
        if ref in by_label and label in by_label:
            tables['flags-tokens-' + label] = report.run('flags-tokens-%s-vs-%s' % (label, ref),
                                                         lambda l=label: check_flag_tokens(root / ref, root / l))
            tables['flags-selection-' + label] = report.run('flags-selection-%s-vs-%s' % (label, ref),
                                                            lambda l=label: check_flag_selection(root / ref, root / l))
            if by_label[label].get('policy') == 'goal_query':
                tables['flags-goal-' + label] = report.run('flags-goal-%s-vs-%s' % (label, ref),
                                                           lambda l=label: check_flag_goals(root / ref, root / l))
            if 'capacity-waits' in wanted:
                overlapped = OVERLAP_PREFIX in ' '.join(runs_by.get(label, {}).get('admission') or ())
                tables['capacity-waits-' + label] = report.run(
                    'capacity-waits-%s-vs-%s' % (label, ref),
                    lambda l=label, o=overlapped: check_capacity_waits(root / ref, root / l, overlap=o))
    if 'adm-report' in wanted and by_label:
        tables['adm-report'] = report.run('adm-report', lambda: adm_report(by_label, options, pins))

    def compared():
        skipped = [r for r in results if r.get('label') in labels and r.get('status') == 'SKIPPED_TIME_BUDGET']
        if not by_label and options['runs'] and all(r['optional'] for r in options['runs']) \
                and len(skipped) == len(options['runs']):
            return dict(ran=[], skipped_optional=sorted(r['label'] for r in skipped))  # an optional segment left out
        need(by_label, 'no run of the segment ran')
        return dict(ran=sorted(by_label))
    report.run('runs-compared', compared)
    # perf-ctx: run A's turns at long context (ITL, admission, restore bytes; the long reply and the probe) and every
    # run's host memory next to the launcher's model.
    if 'context' in wanted and by_label:
        tables['long-context'] = report.run('long-context', lambda: check_long_context(by_label))
    if 'host-memory' in wanted and by_label:
        tables['host-memory'] = report.run('host-memory', lambda: check_host_memory(by_label, pins, options))
    report.run('runs-compared', lambda: (need(by_label, 'no run of the segment ran'), dict(ran=sorted(by_label)))[-1])
    return dict(status='PASS' if not report.failed else 'FAIL', checks=report.checks, failed=report.failed,
                findings=report.findings, tables=tables)


# ---------------------------------------------------------------- perf-adm: the flag runs against off

PARITY_FINDING = 'EAGER_REPEAT_DIFFERS'
FLAG_FINDINGS = {'GRAPH_TOKENS_DIFFER': 'FLAG_TOKENS_DIFFER',
                 'SELECTION_DIFFERS': 'FLAG_SELECTION_DIFFERS', 'ACTIVE_NOT_COMPARED': 'FLAG_ACTIVE_NOT_COMPARED',
                 'PAGE_VECTORS_DIFFER': 'FLAG_VECTORS_DIFFER'}


def dequantized(value):
    """A Page vector as floats: compact_int8_v1 [[q], scale] -> q x scale; a legacy float row as it is."""
    if compact_vector(value):
        return [q * value[1] for q in value[0]]
    return list(value)


def quantiles(values):
    """Nearest-rank quantiles of per-Page cosines (min, p01, p05, p50, max, how many below 1)."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]
    return dict(n=len(values), min=values[0], p01=pick(0.01), p05=pick(0.05), p50=pick(0.5), max=values[-1],
                below_one=sum(1 for v in values if v < 1.0))


def page_cosines(x, y):
    """Per-Page cosine of the common Pages of two {page: vector} maps (dequantized), rounded to 1e-9."""
    out = []
    for page in sorted(set(x) & set(y), key=int):
        value = cosine(dequantized(x[page]), dequantized(y[page]))
        out.append(None if value is None else round(value, 9))
    return out


def compare_runs(reference, other):
    """Two runs of the same tasks turn by turn (session by session, until its prompts diverge): prompt token ids, reply
    and completion count, the selection (reason, candidates, the selected Pages), the observation query hash and row,
    the phase-1 and published int8 Page vectors (identical Pages, per-Page cosines). Returns the per-turn rows and the
    lists of what differed."""
    a, b = run_turns(reference), run_turns(other)
    rows, stopped = [], {}
    differs = dict(tokens=[], selection=[], vectors=[], observation=[], query_hash=[])
    cos1, cos2 = [], []
    for key in sorted(set(a) & set(b)):
        if key[0] in stopped:
            continue
        x, y = a[key], b[key]
        position = first_divergence(x['prompt'], y['prompt'])
        if position is not None:
            stopped[key[0]] = dict(turn=key[1], where='prompt', position=position)
            differs['tokens'].append(list(key))
            continue
        reply = x['content'] == y['content'] and x['completion'] == y['completion']
        sx, sy = list(x['native']['plan']['selected']), list(y['native']['plan']['selected'])
        bx = ((x['native'].get('query') or {}).get('body')) or {}
        by_ = ((y['native'].get('query') or {}).get('body')) or {}
        selection = (selection_reason(x['native']) == selection_reason(y['native']) and sx == sy
                     and (bx.get('selection') or {}).get('candidate_pages') == (by_.get('selection') or {}).get('candidate_pages'))
        px, ox = phase1_vectors(x['native'])
        py, oy = phase1_vectors(y['native'])
        p1, pub = compare_vectors(px, py), compare_vectors(published_vectors(x['native']), published_vectors(y['native']))
        c1 = page_cosines(px, py)
        c2 = page_cosines(published_vectors(x['native']), published_vectors(y['native']))
        cos1 += c1
        cos2 += c2
        vectors = all(v['identical'] == v['pages'] and not v['missing'] for v in (p1, pub))
        row = dict(session=key[0], turn=key[1], reply_equal=reply, completion=[x['completion'], y['completion']],
                   reason=selection_reason(x['native']) or 'active', selection_equal=selection, selected=len(sx),
                   jaccard=round(len(set(sx) & set(sy)) / len(set(sx) | set(sy)), 6) if set(sx) | set(sy) else 1.0,
                   phase1=p1, published=pub, phase1_cosine=quantiles(c1),
                   observation_equal=ox == oy, observation_cosine=None if ox is None or oy is None else round(cosine(ox, oy), 9),
                   query_hash_equal=bx.get('query_hash') == by_.get('query_hash'))
        rows.append(row)
        if not selection:
            differs['selection'].append(list(key))
        if not vectors:
            differs['vectors'].append(list(key))
        if not row['observation_equal']:
            differs['observation'].append(list(key))
        if not row['query_hash_equal']:
            differs['query_hash'].append(list(key))
        if not reply:
            stopped[key[0]] = dict(turn=key[1], where='reply', completion=[x['completion'], y['completion']],
                                   position=first_divergence(generated_ids(a, key) or [], generated_ids(b, key) or [])
                                   if generated_ids(a, key) is not None and generated_ids(b, key) is not None else None)
            differs['tokens'].append(list(key))
    return dict(reference=Path(reference).name, other=Path(other).name, turns=rows,
                turns_reference=len(a), turns_other=len(b), turns_compared=len(rows),
                identical_turns=sum(1 for r in rows if r['reply_equal'] and r['selection_equal']
                                    and r['phase1']['identical'] == r['phase1']['pages']
                                    and r['published']['identical'] == r['published']['pages']),
                active_turns=sum(1 for r in rows if r['reason'] == 'active'),
                first_divergence={str(k): v for k, v in sorted(stopped.items())}, differs=differs,
                phase1_pages=sum(r['phase1']['pages'] for r in rows),
                phase1_identical_pages=sum(r['phase1']['identical'] for r in rows),
                published_pages=sum(r['published']['pages'] for r in rows),
                published_identical_pages=sum(r['published']['identical'] for r in rows),
                phase1_cosine=quantiles(cos1), published_cosine=quantiles(cos2))


def check_parity(reference, other):
    """S = 1 parity (a failure otherwise): `other` (a flag run: e, b or all) completed the reference's turns, and on
    every turn its prompt token ids, reply, selection (reason, candidates, selected Pages) and int8 Page vectors (the
    phase-1 receipt's and the publication's, every Page bit for bit) equal the flag-off reference's. The observation
    query row and hash are reported (they follow the same hidden rows). The detail carries the per-Page cosines."""
    detail = compare_runs(reference, other)
    for kind in ('tokens', 'selection', 'vectors'):
        need(not detail['differs'][kind], '%s differs from %s in %s at %s (first divergence %s; phase-1 cosine %s)'
             % (detail['other'], detail['reference'], kind, detail['differs'][kind][:4], detail['first_divergence'],
                detail['phase1_cosine']))
    need(detail['turns_reference'] == detail['turns_other'] == detail['turns_compared'],
         '%s completed %d turns, %s %d, %d compared' % (detail['other'], detail['turns_other'], detail['reference'],
                                                        detail['turns_reference'], detail['turns_compared']))
    return detail


def check_repeat(reference, other):
    """The eager repeat control (off again against off, one session): is the long eager prefill itself deterministic?
    Any difference in tokens, selections, Page vectors or the observation row is the finding EAGER_REPEAT_DIFFERS
    (never a failure: it says how far the parity checks can be trusted); the per-Page cosine quantiles either way."""
    detail = compare_runs(reference, other)
    if any(detail['differs'][k] for k in ('tokens', 'selection', 'vectors', 'observation')) or \
            detail['turns_reference'] != detail['turns_other']:
        raise Finding(PARITY_FINDING, detail)
    return detail


def flag_finding(function, reference, other):
    try:
        return function(reference, other)
    except Finding as exc:
        raise Finding(FLAG_FINDINGS.get(exc.code, exc.code), exc.detail)


def check_flag_tokens(reference, other):
    """S > 1: a flag run's tokens against off's, session by session (compare_tokens). Batch composition depends on
    timing, so a divergence is the finding FLAG_TOKENS_DIFFER, never a failure."""
    return flag_finding(compare_tokens, reference, other)


def check_flag_selection(reference, other):
    """S > 1: a flag run's selections and Page vectors against off's while the prompts agree (compare_selection; mixed
    steps change the low bits of phase-1 vectors): findings FLAG_SELECTION_DIFFERS, FLAG_ACTIVE_NOT_COMPARED,
    FLAG_VECTORS_DIFFER; the per-Page cosine quantiles of the compared turns."""
    def compared(a, b):
        try:
            detail = compare_selection(a, b)
        except Finding as exc:
            exc.detail.update(phase1_cosine=selection_cosines(a, b, exc.detail))
            raise
        detail.update(phase1_cosine=selection_cosines(a, b, detail))
        return detail
    return flag_finding(compared, reference, other)


def selection_cosines(reference, other, detail):
    """Per-Page cosine quantiles of the phase-1 vectors of the turns compare_selection compared."""
    a, b = run_turns(reference), run_turns(other)
    values = []
    for row in detail.get('turns') or ():
        key = (row['session'], row['turn'])
        values += page_cosines(phase1_vectors(a[key]['native'])[0], phase1_vectors(b[key]['native'])[0])
    return quantiles(values)


def capacity_waits(out):
    """CAPACITY_WAIT refusals of one server: its rows' and failures' capacity_waits."""
    out = Path(out)
    return (sum(r.get('capacity_waits') or 0 for r in measure.runtime_rows(out))
            + sum(f.get('capacity_waits') or 0 for f in failure_rows(out)))


OVERLAP_PREFIX = 'admission_overlap='
OVERLAP_CAPACITY_FINDING = 'OVERLAP_CAPACITY_WAITS'
GOAL_FINDING = 'FLAG_GOAL_DIFFERS'


def check_capacity_waits(reference, other, overlap=False):
    """A flag run never waits for capacity more often than off (the (e) gate counts the held probe slot: three GDN
    slots in phase 2, K = 2S + 1 unchanged). (f) `overlap`: every admission in flight holds a reservation in the
    capacity ledger (with (e) three GDN slots), so more CAPACITY_WAITs than off are the finding OVERLAP_CAPACITY_WAITS,
    not a failure."""
    mine, base = capacity_waits(other), capacity_waits(reference)
    detail = dict(label=Path(other).name, capacity_waits=mine, reference=Path(reference).name,
                  reference_capacity_waits=base, overlap=overlap)
    if overlap:
        if mine > base:
            raise Finding(OVERLAP_CAPACITY_FINDING, detail)
        return detail
    need(mine <= base, '%s: %d CAPACITY_WAITs, %s %d' % (Path(other).name, mine, Path(reference).name, base))
    return detail


GOAL_KEYS = ('gate_decision', 'applied_decision', 'slm_attempted', 'slm_succeeded', 'fallback_reason', 'variant',
             'payload_sha256')


def check_flag_goals(reference, other):
    """(f) goal_query: a flag run's Goals against off's, session by session while the prompts coincide (the Goal is a
    function of its turn's context, its session's constructor state and the rewriter's greedy output; concurrent and
    serial constructions give equal Goal records): the Goal decisions (gate and applied), the rewriter use, the
    fallback, the Goal payload (payload_sha256: the retrieval Goal's text) and goal_hash. Any difference is the
    finding FLAG_GOAL_DIFFERS (the rewriter shares the GPU with mixed batches), never a failure."""
    a, b = run_turns(reference), run_turns(other)
    rows, stopped, differs = [], set(), []
    for key in sorted(set(a) & set(b)):
        if key[0] in stopped:
            continue
        if first_divergence(a[key]['prompt'], b[key]['prompt']) is not None:
            stopped.add(key[0])
            continue
        x = ((a[key]['native'].get('query') or {}).get('body')) or {}
        y = ((b[key]['native'].get('query') or {}).get('body')) or {}
        gx, gy = x.get('goal') or {}, y.get('goal') or {}
        fields = [k for k in GOAL_KEYS if gx.get(k) != gy.get(k)] + (['goal_hash'] if x.get('goal_hash') != y.get('goal_hash') else [])
        rows.append(dict(session=key[0], turn=key[1], equal=not fields, differs=fields,
                         applied=[gx.get('applied_decision'), gy.get('applied_decision')]))
        if fields:
            differs.append(list(key))
        if a[key]['content'] != b[key]['content']:
            stopped.add(key[0])  # the next prompts differ
    detail = dict(reference=Path(reference).name, other=Path(other).name, turns_compared=len(rows),
                  equal_turns=sum(1 for r in rows if r['equal']), differs=differs, turns=rows)
    if differs:
        raise Finding(GOAL_FINDING, detail)
    return detail


ADM_REPORT_KEYS = ('mixed_share', 'mixed', 'parked', 'reparked', 'step_excess_s', 'fallback_phase1_s',
                   'stall_with_steps_fraction')


def overlap_columns(overlap):
    """(f) columns of the admission report from measure.overlap_summary."""
    pick = lambda name, key: (overlap.get(name) or {}).get(key)
    return dict(queue_wait_s_mean=pick('queue_wait_s', 'mean'), queue_wait_s_p95=pick('queue_wait_s', 'p95'),
                queue_wait_share=overlap.get('queue_wait_share'),
                goal_async_ms_mean=pick('goal_async_ms', 'mean'), goal_async_ms_p95=pick('goal_async_ms', 'p95'),
                goal_construct_ms_mean=pick('goal_construct_ms', 'mean'),
                lease_waits=overlap.get('lease_waits'), lease_wait_ms_mean=pick('lease_wait_ms', 'mean'),
                lease_wait_ms_max=pick('lease_wait_ms', 'max'),
                admission_engine_ms_p95=pick('admission_engine_ms', 'p95'),
                rewriter_calls=overlap.get('rewriter_calls'), rewriter_busy_s=overlap.get('rewriter_busy_s'),
                rewriter_busy_share=overlap.get('rewriter_busy_share'))


def adm_report(by_label, options, pins):
    """The admission report of a segment (informational, never a finding): per run its flags and graph mode, the stall
    share (admission_engine_ms over the span) and with the probe steps' excess and parked fallbacks counted
    (measure.phase1_summary), how phase 1 ran (modes, mixed share, parked), the step that carried a probe against a plain
    decode step at the same batch size without it, ITL and tok/s at the server's sessions, generated tok/s, the
    admission (engine thread) and phase-1 means, restore H2D and D2D bytes per continuing turn, CAPACITY_WAITs and
    vLLM's preemptions; next to the pinned reference numbers (pins.json reference_numbers: job 3343069's stall share)."""
    table = []
    for label, summary in by_label.items():
        section = summary.get('measurements') or {}
        aggregate, batch, phase1 = section.get('aggregate') or {}, section.get('batch') or {}, section.get('phase1') or {}
        sessions = summary.get('running') or summary.get('sessions')
        top = {b['batch']: b for b in batch.get('batches') or ()}.get(sessions) or {}
        turns = [t for t in summary.get('turns') or () if t.get('turn') and type(t.get('admission_engine_ms')) in (int, float)]
        rows = [r for r in section.get('rows') or () if r.get('valid') and r.get('turn')]
        mean = lambda values: round(sum(values) / len(values), 3) if values else None
        table.append(dict(
            segment=options['segment'], label=label, admission=summary.get('admission') or [],
            cuda_graphs=summary.get('cuda_graphs'),
            sessions=sessions, requests=aggregate.get('requests'), span_s=aggregate.get('span_s'),
            stall_share=aggregate.get('admission_stall_fraction'), stall_s=aggregate.get('admission_stall_s'),
            **{k: phase1.get(k) for k in ADM_REPORT_KEYS},
            modes=phase1.get('modes'), continuing_turns=phase1.get('continuing_turns'),
            step_ms=phase1.get('step_ms'), step_rows=phase1.get('step_rows'),
            plain_itl_ms_at_rows=phase1.get('plain_itl_ms_at_rows'), step_excess_ms=phase1.get('step_excess_ms'),
            park_ms=phase1.get('park_ms'), steady_itl_ms_at_sessions=top.get('steady_itl_ms'),
            steady_tok_s_at_sessions=top.get('steady_tok_s'), gen_tokens_per_s=aggregate.get('gen_tokens_per_s'),
            turns_per_min=aggregate.get('turns_per_min'), running_peak=batch.get('running_peak'),
            admission_engine_ms_mean=mean([t['admission_engine_ms'] for t in turns]),
            phase1_ms_mean=mean([r['phase1_ms'] for r in rows if r.get('phase1_ms') is not None]),
            restore_h2d_bytes_mean=mean([r['h2d_other_bytes'] for r in rows if r.get('h2d_other_bytes') is not None]),
            d2d_bytes_mean=mean([r['d2d_bytes'] for r in rows if r.get('d2d_bytes') is not None]),
            capacity_waits=aggregate.get('capacity_waits'), preemptions_max=batch.get('preemptions_max'),
            **overlap_columns(section.get('overlap') or {})))
    reference = pins.get('reference_numbers') or {}
    return dict(segment=options['segment'], table=table,
                reference=dict(job_id=reference.get('job_id'), runs=reference.get('runs')) if reference else None)


# ---------------------------------------------------------------- perf-int: CUDA graphs (perf-cudagraph-rt, scg2)

# The runtime's pinned vLLM compilation config per --cuda-graphs mode (remote_ra_live_v2/cuda_graphs.py of the
# cudagraph branch; controller.COMPILATION), the ops that stay outside every compiled graph, and the vLLM 0.24.0
# server-log lines of each mode (scg1 job 3330868's stock logs, scg2 job 3338893's runtime logs).
COMPILATION = {'piecewise': dict(cudagraph_mode='PIECEWISE', mode='VLLM_COMPILE'),
               'full': dict(cudagraph_mode='FULL_AND_PIECEWISE', mode='VLLM_COMPILE'),
               'full-decode': dict(cudagraph_mode='FULL_DECODE_ONLY', mode='NONE')}
REQUIRED_SPLITTING_OPS = ['vllm::unified_attention_with_output', 'vllm::unified_kv_cache_update',
                          'vllm::qwen_gdn_attention_core']
CAPTURE_PIECEWISE = 'Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)'
CAPTURE_FULL = 'Capturing CUDA graphs (decode, FULL)'
CAPTURE_DONE = 'Graph capturing finished in'
EAGER_LINE = 'Enforce eager set, disabling torch.compile and CUDAGraphs'
COMPILE_LINES = ('Compiling a graph for compile range', 'Directly load the compiled graph')
TORCH_COMPILE_LINE = 'torch.compile took'
CONFIG_FINDING = 'CUDA_GRAPH_CONFIG_DIFFERS'
GRAPH_FINDING = 'GRAPH_TOKENS_DIFFER'
SELECTION_FINDING = 'SELECTION_DIFFERS'
VECTOR_FINDING = 'PAGE_VECTORS_DIFFER'
ACTIVE_COMPARED_FINDING = 'ACTIVE_NOT_COMPARED'
ACTIVE_BATCH_FINDING = 'ACTIVE_NOT_BATCHED'
# A capture progress bar's last state: "Capturing CUDA graphs (decode, FULL): 100%|...| 5/5".
CAPTURE_COUNT = re.compile(r'Capturing CUDA graphs \(([^)]*)\):[^\n]*?(\d+)/(\d+)')
ENGINE_CONFIG = re.compile(r'Initializing a V1 LLM engine \(v[0-9.]+\) with config: (.*)')
CONFIG_PATTERNS = dict(
    enforce_eager=(re.compile(r'\benforce_eager=(True|False)'), lambda v: v == 'True'),
    compilation_mode=(re.compile(r"'mode': <CompilationMode\.([A-Z_]+): [^>]*>"), str),
    cudagraph_mode=(re.compile(r"'cudagraph_mode': <CUDAGraphMode\.([A-Z_]+): [^>]*>"), str),
    cudagraph_capture_sizes=(re.compile(r"'cudagraph_capture_sizes': \[([0-9, ]*)\]"),
                             lambda v: [int(x) for x in v.replace(' ', '').split(',') if x]),
    max_cudagraph_capture_size=(re.compile(r"'max_cudagraph_capture_size': (\d+|None)"),
                                lambda v: None if v == 'None' else int(v)))
# Startup lines (vLLM 0.24.0; perf-cudagraph-rt stock.STARTUP_PATTERNS): absent -> None, never a failure.
STARTUP_PATTERNS = dict(
    model_load_gib=(re.compile(r'Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds'), 1, float),
    model_load_s=(re.compile(r'Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds'), 2, float),
    dynamo_transform_s=(re.compile(r'Dynamo bytecode transform time: ([\d.]+) s'), 1, float),
    torch_compile_s=(re.compile(r'torch\.compile (?:takes|took) ([\d.]+) s in total'), 1, float),
    graph_capture_s=(re.compile(r'Graph capturing finished in ([\d.]+) secs?'), 1, float),
    graph_capture_gib=(re.compile(r'Graph capturing finished in [\d.]+ secs?, took ([\d.]+) GiB'), 1, float),
    init_engine_s=(re.compile(r'init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s'), 1, float))
COMPILE_HIT = re.compile(r'Directly load the compiled graph')
COMPILE_MISS = re.compile(r'Compiling a graph for')


def engine_config(text):
    """The resolved compilation fields of the engine config line (None when a field or the line is absent)."""
    line = ENGINE_CONFIG.search(text)
    config = line.group(1) if line else ''
    out = {}
    for key, (pattern, cast) in CONFIG_PATTERNS.items():
        match = pattern.search(config)
        out[key] = cast(match.group(1)) if match else None
    out['config_line'] = bool(line)
    return out


def capture_counts(text):
    """{capture kind: (graphs captured, of) at its last progress state} of a server log."""
    counts = {}
    for kind, done, total in CAPTURE_COUNT.findall(text):
        counts[kind] = (int(done), int(total))
    return counts


# The engines of one server log (perf-swe32's smoke_checks.engine_logs, ported 2026-11-03 for the release smoke plans;
# perf-swe32's job 3500565, vllm25 r2). serve.py --goal-rewriter-engine vllm (integrate-1030, rewriter-lfe-fix c07d42a,
# in integrate-1103) runs the Goal rewriter as a second vLLM engine: a child process whose own lines carry no process
# prefix (vLLM's config lines, "goal rewriter vllm child: ...") and whose EngineCore ("(EngineCore pid=N)") logs into
# the same server log as the main engine. 3500565: the main EngineCore pid 15428 (max_seq_len 262144, capture sizes
# 1..32, PIECEWISE 7/7, FULL 5/5), then the child's pid 33376 (max_seq_len 16384, sizes 1..16, PIECEWISE 5/5, FULL 5/5;
# FLASH_ATTN with FlashAttention 3, its own "Asynchronous scheduling is enabled." lines, one of them without a prefix);
# read as text, the tqdm capture bars' carriage returns end lines, so the bars' states carry no prefix. context-limits,
# cuda-graphs, capture-sizes, kernel-alignment and async-scheduling read the main engine's lines only; the child's own
# config is information in their detail.
PROCESS_TAG = re.compile(r'\(([A-Za-z][A-Za-z0-9_]*) pid=(\d+)\)')
ENGINE_MODEL = re.compile(r"\bmodel='([^']*)'")
ENGINE_STARTED = 'init engine (profile, create kv cache, warmup model) took'
# what the engine-side checks read: a line with one of these and no process prefix must be attributed
ENGINE_SIDE_LINES = ('Initializing a V1 LLM engine', 'max_seq_len=', 'Capturing CUDA graphs', CAPTURE_DONE, EAGER_LINE,
                     TORCH_COMPILE_LINE) + COMPILE_LINES


def rewriter_engine_model(out):
    """The vLLM Goal rewriter child's checkpoint: startup.json's goal_rewriter_engine record (engine vllm)
    engine_args.model, else None (no such child)."""
    path = Path(out) / 'startup.json'
    record = load(path).get('goal_rewriter_engine') if path.is_file() else None
    args = record.get('engine_args') if isinstance(record, dict) and record.get('engine') == 'vllm' else None
    model = args.get('model') if isinstance(args, dict) else None
    return model if type(model) is str and model else None


def engine_logs(out, text):
    """A server log's lines per vLLM engine: dict(main=text, rewriter=[dict(pid, model, text)], split=bool,
    main_pid=the main EngineCore's pid or None, tags=the main engine's process tags).

    One engine (at most one engine config line): the whole log is the main engine's, exactly as before (every real
    server log before job 3500565 has one). Otherwise every "Initializing a V1 LLM engine" config line is one engine:
    the Goal rewriter child's names the checkpoint of startup.json's goal_rewriter_engine record (engine vllm), and
    exactly one other engine, the main one, must exist; a further engine, an EngineCore pid without a config line or
    with two fail. A line with an EngineCore prefix belongs to that pid's engine and an APIServer line to the main
    server; any other line the engine-side checks read (ENGINE_SIDE_LINES: vLLM's config, compile and capture lines,
    the capture bars' states) belongs to the one engine whose startup, from its config line to its own "init engine ...
    took" line (else the log's end), contains it: in none or in two it cannot be attributed and fails.
    main = the log without the child engines' lines."""
    lines = text.split('\n')
    tags, configs, pids = [], [], []
    for i, line in enumerate(lines):
        tag = PROCESS_TAG.match(line)
        tag = (tag.group(1), int(tag.group(2))) if tag else None
        tags.append(tag)
        pid = tag[1] if tag and tag[0].startswith('EngineCore') else None
        if pid is not None and pid not in pids:
            pids.append(pid)
        config = ENGINE_CONFIG.search(line)
        if config:
            model = ENGINE_MODEL.search(config.group(1))
            configs.append(dict(line=i, pid=pid, model=model.group(1) if model else None))
    if len(configs) <= 1:
        return dict(main=text, rewriter=[], split=False, main_pid=configs[0]['pid'] if configs else None)
    child = rewriter_engine_model(out)
    for c in configs:
        c['rewriter'] = child is not None and c['model'] == child
    named = [dict(pid=c['pid'], model=c['model'], line=c['line'] + 1) for c in configs]
    main = [c for c in configs if not c['rewriter']]
    need(len(main) == 1, 'the server log has %d vLLM engines that are not the Goal rewriter child (the engine on '
         'startup.json\'s goal_rewriter_engine checkpoint %r), expected exactly one, the main engine: %s'
         % (len(main), child, named))
    main = main[0]
    tagged = [c['pid'] for c in configs if c['pid'] is not None]
    need(len(tagged) == len(set(tagged)), 'EngineCore pids with more than one engine config line: %s (engines %s)'
         % (sorted({p for p in tagged if tagged.count(p) > 1}), named))
    unknown = [p for p in pids if p not in tagged]
    need(not unknown, 'EngineCore pid(s) %s log lines but no engine config line: their lines cannot be attributed to '
         'the main engine or the Goal rewriter child (engines %s)' % (unknown, named))
    for c in configs:
        c['end'] = next((j for j in range(c['line'] + 1, len(lines)) if ENGINE_STARTED in lines[j] and (
            tags[j][1] if tags[j] and tags[j][0].startswith('EngineCore') else None) == c['pid']), len(lines))
    by_pid = {c['pid']: c for c in configs if c['pid'] is not None}
    own = {id(c): [] for c in configs}
    rest, unattributed = [], []
    for j, line in enumerate(lines):
        tag = tags[j]
        if tag and tag[0].startswith('EngineCore'):
            owner = by_pid[tag[1]]
        elif tag and tag[0] == 'APIServer':
            owner = main
        elif any(marker in line for marker in ENGINE_SIDE_LINES):
            within = [c for c in configs if c['line'] <= j <= c['end']]
            if len(within) != 1:
                unattributed.append(dict(line=j + 1, engines=[c['line'] + 1 for c in within]))
                continue
            owner = within[0]
        else:
            rest.append(line)  # read by no engine-side check (the child's own lines, C++ warnings): the main text's
            continue
        own[id(owner)].append(line)
        if owner is main:
            rest.append(line)
    need(not unattributed, '%d server-log lines of the engine-side checks (config, compile, capture lines) lie in no '
         'engine\'s startup or in two, so they cannot be attributed to the main engine or the Goal rewriter child: %s '
         '(engines %s; line numbers of the log read as text, where a carriage return ends a line)'
         % (len(unattributed), unattributed[:10], named))
    return dict(main='\n'.join(rest), split=True, main_pid=main['pid'], rewriter=[
        dict(pid=c['pid'], model=c['model'], text='\n'.join(own[id(c)])) for c in configs if c['rewriter']])


def main_engine_tagged(engines, text):
    """The lines of `text` that carry the main engine's own process tag: its APIServer and its EngineCore pid (one
    engine: every APIServer and EngineCore line). vLLM's config summaries (the async scheduling line) are logged by
    the API server and the EngineCore of each engine; the vLLM Goal rewriter child logs its own, one of them without
    a process prefix (job 3500565), so a check of the main engine's config reads its tagged lines only."""
    out = []
    for line in text.split('\n'):
        tag = PROCESS_TAG.match(line)
        if not tag:
            continue
        name, pid = tag.group(1), int(tag.group(2))
        if name == 'APIServer' or (name.startswith('EngineCore') and (
                not engines['split'] or engines.get('main_pid') is None or pid == engines['main_pid'])):
            out.append(line)
    return out


def rewriter_engine_info(engines):
    """The Goal rewriter child engines' own startup as their lines show it (information, never checked): pid,
    checkpoint, max_seq_len, capture sizes and capture counts."""
    info = []
    for e in engines['rewriter']:
        config = engine_config(e['text'])
        info.append(dict(pid=e['pid'], model=e['model'],
                         max_seq_len=sorted({int(v) for v in MAX_SEQ_LEN.findall(e['text'])}),
                         capture_sizes=config['cudagraph_capture_sizes'],
                         capture_counts={k: list(v) for k, v in capture_counts(e['text']).items()}))
    return info


def startup_info(log_text):
    """What the server log says about its startup, compilation and CUDA graphs (model load, torch.compile, capture time
    and memory, compile-cache hits, capture kinds and counts, the engine config's compilation fields); the verifier
    recomputes it from the archived log."""
    out = {}
    for key, (pattern, group, cast) in STARTUP_PATTERNS.items():
        match = pattern.search(log_text)
        out[key] = cast(match.group(group)) if match else None
    out.update({k: v for k, v in engine_config(log_text).items() if k != 'config_line'})
    out['capture_counts'] = {k: list(v) for k, v in capture_counts(log_text).items()}
    out['compile_cache_hits'] = len(COMPILE_HIT.findall(log_text))
    out['compile_cache_misses'] = len(COMPILE_MISS.findall(log_text))
    out['eager_line'] = EAGER_LINE in log_text
    return out


def capture_sizes(sessions, max_batched=MAX_BATCHED):
    """vLLM 0.24.0's default CUDA-graph capture sizes of a server with `sessions` sequences (vllm/config/vllm.py
    _set_cudagraph_sizes, no speculative decoding): 1, 2, 4 up to the largest size min(2 x max_num_seqs, 512,
    max_num_batched_tokens), then steps of 8 below 256 (16 from 256). scg2 job 3338893: [1, 2, 4, 8, 16] at 8 sequences."""
    largest = min(sessions * 2, 512, max_batched)
    sizes = [i for i in (1, 2, 4) if i <= largest]
    if largest >= 8:
        sizes += list(range(8, min(largest + 1, 256), 8))
    if largest >= 256:
        sizes += list(range(256, largest + 1, 16))
    if max_batched <= largest and max_batched not in sizes:
        sizes.append(max_batched)
    return sorted(set(sizes))


def check_cuda_graphs(out, expect, server_log):
    """The server ran exactly its layout's CUDA-graph mode (the cudagraph branch's `--cuda-graphs`), recorded once at
    startup, with a CUDA graph for its full batch (perf-int: 16 sessions decode in one FULL graph step).

    off: startup.json has no `cuda_graphs`, the effective config pins enforce_eager true and no compilation config,
    worker-attached.json has no `cuda_graphs` record, the log shows vLLM's eager line and neither compilation nor
    capture. A graph mode: startup `cuda_graphs` = the mode, the effective config enforce_eager false with the pinned
    compilation config; the worker record of the resolved configuration names the mode, its compilation and
    CUDA-graph mode, the three ops outside the compiled graphs (compiled modes), every FullAttention group bound to the
    physical seq-len buffer in the FULL modes (none in piecewise) and phase 1 without graphs; the engine config line
    resolves the same compilation and CUDA-graph mode; the capture sizes (worker record and engine config line) reach
    the server's max_num_seqs S (perf-adm: a size >= S; 16 itself at S = 16), so a decode step of all S sessions runs as
    a graph; the log shows the mode's captures
    completed (piecewise and full capture mixed steps piecewise, full and full-decode capture decode steps whole), its
    compilation (or the compile cache) exactly in the compiled modes, and no eager line. These are checks. What vLLM
    chooses itself beyond that (the capture size list and counts, the Inductor backend, custom ops `none` compiled and
    `all` otherwise, the torch.compile total line) is compared with vLLM 0.24.0's defaults: a difference is the finding
    CUDA_GRAPH_CONFIG_DIFFERS, never a failure. With a vLLM Goal rewriter child every log reading is the main
    engine's (engine_logs); the child's own startup is information (`rewriter_engines`)."""
    layout = layout_of(expect)
    mode = layout.get('cuda_graphs', 'off')
    startup, ready = load(out / 'startup.json'), load(out / 'ready.json')
    attached = load(out / 'worker-attached.json')
    effective = ready.get('effective_server_config') or {}
    record = attached['receipt'].get('cuda_graphs')
    engines = engine_logs(out, Path(server_log).read_text(encoding='utf-8', errors='replace'))
    text = engines['main']
    child = dict(rewriter_engines=rewriter_engine_info(engines)) if engines['split'] else {}
    seen = dict(piecewise=CAPTURE_PIECEWISE in text, full=CAPTURE_FULL in text, done=CAPTURE_DONE in text,
                eager=EAGER_LINE in text, compiled=any(line in text for line in COMPILE_LINES))
    if mode == 'off':
        need('cuda_graphs' not in startup and effective.get('enforce_eager') is True
             and 'compilation_config' not in effective, 'eager server pins %r / enforce_eager %r / compilation %r'
             % (startup.get('cuda_graphs'), effective.get('enforce_eager'), effective.get('compilation_config')))
        need(record is None, 'eager server records CUDA graphs %r' % (record,))
        need(seen['eager'] and not (seen['piecewise'] or seen['full'] or seen['done'] or seen['compiled']),
             'eager server log %s' % seen)
        return dict(mode='off', log=seen, **child)
    wanted = COMPILATION[mode]
    need(startup.get('cuda_graphs') == mode and effective.get('enforce_eager') is False
         and effective.get('compilation_config') == wanted,
         'startup cuda_graphs %r / enforce_eager %r / compilation %r, expected %s / False / %r'
         % (startup.get('cuda_graphs'), effective.get('enforce_eager'), effective.get('compilation_config'), mode, wanted))
    need(isinstance(record, dict), 'worker records no cuda_graphs (%r)' % (record,))
    attention = sorted({v['group'] for v in attached['receipt'].get('tensor_geometry') or () if v.get('kind') == 'attention'})
    need(attention, 'worker tensor geometry names no FullAttention group')
    compiled = wanted['mode'] == 'VLLM_COMPILE'
    bound = record.get('seq_lens_bound_groups')
    sessions = expect.get('max_num_seqs') or len(expect['tasks'])
    # the window layout (Gemma) keeps attention and the KV-cache update outside the compiled pieces (no GDN op:
    # cuda_graphs.WINDOW_SPLITTING_OPS)
    ops = list(REQUIRED_SPLITTING_OPS[:2]) if layout.get('window') else list(REQUIRED_SPLITTING_OPS)
    checked = dict(mode=record.get('mode') == mode, compilation_mode=record.get('compilation_mode') == wanted['mode'],
                   cudagraph_mode=record.get('cudagraph_mode') == wanted['cudagraph_mode'],
                   splitting_ops=not compiled or record.get('splitting_ops') == ops,
                   seq_lens_bound_groups=(isinstance(bound, list) and set(attention) <= set(bound))
                   if mode in ('full', 'full-decode') else bound == [],
                   probe_forward=record.get('probe_forward') == 'eager',
                   capture_covers_sessions=any(s >= sessions for s in record.get('capture_sizes') or ()))
    need(all(checked.values()), 'worker cuda_graphs record %r fails %s (attention groups %s, %d sessions)'
         % (record, sorted(k for k, v in checked.items() if not v), attention, sessions))
    logs = dict(piecewise=mode in ('piecewise', 'full'), full=mode in ('full', 'full-decode'), done=True, eager=False,
                compiled=compiled)
    need(seen == logs, '%s server log %s, expected %s' % (mode, seen, logs))
    config = engine_config(text)
    need(config['config_line'] and config['enforce_eager'] is False and config['compilation_mode'] == wanted['mode']
         and config['cudagraph_mode'] == wanted['cudagraph_mode'],
         'engine config line: enforce_eager %r, mode %r, cudagraph_mode %r; expected False, %s, %s'
         % (config['enforce_eager'], config['compilation_mode'], config['cudagraph_mode'], wanted['mode'],
            wanted['cudagraph_mode']))
    sizes = config['cudagraph_capture_sizes'] or []
    # A batch of S runs as the graph of the smallest capture size >= S (vLLM pads): 16 is a size at S = 16; at S = 5
    # the full batch pads to 8 (a FULL decode graph exists only for sizes <= max_num_seqs, so batch 5 runs piecewise).
    need(any(s >= sessions for s in sizes), 'engine config capture sizes %s do not cover the %d-session decode step'
         % (sizes, sessions))
    counts = capture_counts(text)
    kinds = (['mixed prefill-decode, PIECEWISE'] if mode in ('piecewise', 'full') else []) + \
        (['decode, FULL'] if mode in ('full', 'full-decode') else [])
    for kind in kinds:
        done, total = counts.get(kind, (0, None))
        need(total and done == total, 'capture %r did not complete in the log: %s' % (kind, counts.get(kind)))
    defaults = capture_sizes(sessions)
    expected_counts = {}
    if mode in ('piecewise', 'full'):
        expected_counts['mixed prefill-decode, PIECEWISE'] = len(defaults)
    if mode in ('full', 'full-decode'):
        expected_counts['decode, FULL'] = len([s for s in defaults if s <= sessions])
    wanted_defaults = dict(capture_sizes=defaults, max_capture_size=defaults[-1], backend='inductor',
                           custom_ops=['none'] if compiled else ['all'], torch_compile_line=compiled,
                           config_capture_sizes=defaults,
                           capture_counts={k: [v, v] for k, v in expected_counts.items()})
    got = dict(capture_sizes=record.get('capture_sizes'), max_capture_size=record.get('max_capture_size'),
               backend=record.get('backend'), custom_ops=record.get('custom_ops'),
               torch_compile_line=TORCH_COMPILE_LINE in text, config_capture_sizes=sizes,
               capture_counts={k: list(counts[k]) for k in expected_counts if k in counts})
    detail = dict(mode=mode, sessions=sessions, record=record, log=seen, attention_groups=attention,
                  engine_config={k: v for k, v in config.items() if k != 'config_line'},
                  capture_counts={k: list(v) for k, v in counts.items()}, defaults=wanted_defaults, **child)
    differ = sorted(k for k in wanted_defaults if got[k] != wanted_defaults[k])
    if differ:
        raise Finding(CONFIG_FINDING, dict(detail, differs={k: dict(got=got[k], default=wanted_defaults[k])
                                                             for k in differ}))
    return detail


def run_turns(out):
    """{(session, turn): dict(prompt, completion, content, native)} of one server, from its runtime rows (by response)."""
    out = Path(out)
    rows = {r['response_id']: r for r in measure.runtime_rows(out)}
    turns, k = {}, 0
    while (out / ('s%d' % k) / 'session.json').exists():
        client = load(out / ('s%d' % k) / 'session.json')
        for t in range(client['completed_turns']):
            response = load(out / ('s%d' % k) / ('http-response-%d.json' % t))
            row = rows.get(response.get('id'))
            need(row is not None, 's%d turn %d of %s has no runtime row' % (k, t, out.name))
            turns[(k, t)] = dict(prompt=row['prompt_token_ids'], content=response['choices'][0]['message'].get('content'),
                                 completion=(response.get('usage') or {}).get('completion_tokens'),
                                 prepare=row['prepare']['h2d_bytes'], restore=row['restore']['h2d_bytes'],
                                 native=load(native_path(out, row['request_id'])))
        k += 1
    return turns


def generated_ids(turns, key):
    """G_t of a turn: its generated token ids, read from the next turn's prompt (None without a next turn)."""
    k, t = key
    following = turns.get((k, t + 1))
    if following is None:
        return None
    prompt, completion = turns[key]['prompt'], turns[key]['completion']
    need(following['prompt'][:len(prompt)] == prompt and type(completion) is int and completion > 0,
         's%d turn %d: the next prompt does not extend this one' % key)
    return following['prompt'][len(prompt):len(prompt) + completion]


def compare_tokens(reference, other):
    """Graph run against the eager run of its layout, session by session: identical prompts (every earlier reply),
    replies and completion counts turn by turn. A divergence is the finding GRAPH_TOKENS_DIFFER (compiled kernels and
    another batch timing change bf16 low bits), never a failure; it names each session's first diverging turn and
    token position (in G_t when the next prompt shows it). A selective run restores the Pages its selection names, so
    H2D is reported, not required."""
    a, b = run_turns(reference), run_turns(other)
    common = sorted(set(a) & set(b))
    need(common, 'no turn completed by both %s and %s' % (Path(reference).name, Path(other).name))
    identical, h2d_equal, firsts = [], [], {}
    for key in common:  # session by session; a session stops at its first divergence
        if key[0] in firsts:
            continue
        x, y = a[key], b[key]
        position = first_divergence(x['prompt'], y['prompt'])
        if position is not None:
            firsts[key[0]] = dict(session=key[0], turn=key[1], where='prompt', position=position)
            continue
        same = x['content'] == y['content'] and x['completion'] == y['completion']
        if not same:
            gx, gy = generated_ids(a, key), generated_ids(b, key)
            firsts[key[0]] = dict(session=key[0], turn=key[1], where='reply', completion=[x['completion'], y['completion']],
                                  position=first_divergence(gx, gy) if gx is not None and gy is not None else None)
            continue
        identical.append(list(key))
        h2d_equal.append((x['prepare'], x['restore']) == (y['prepare'], y['restore']))
    sessions = sorted({k for k, _ in common})
    first = firsts[min(firsts)] if firsts else None
    detail = dict(reference=Path(reference).name, other=Path(other).name, turns_compared=len(common),
                  identical_turns=len(identical), identical=identical, h2d_equal_while_identical=all(h2d_equal),
                  first_divergence=first, sessions=len(sessions), diverged_sessions=sorted(firsts),
                  divergence_by_session=[firsts.get(k) for k in sessions])
    selective = any((t['native'].get('plan') or {}).get('policy') in SELECTIVE for t in a.values())
    if selective:
        detail['h2d'] = 'informational: a selective run restores the Pages its selection names'
    need(selective or all(h2d_equal), 'H2D (prepare, restore) differs on an identical turn: %s' % detail)
    if first is not None:
        raise Finding(GRAPH_FINDING, detail)
    return detail


def phase1_vectors(native):
    """{page: [q, scale]} of a turn's phase-1 frame and its observation query row."""
    for frame in native['worker_frames']:
        if frame['operation'] == 'phase1':
            return dict(frame.get('page_vectors') or {}), frame.get('observation_vector')
    return {}, None


def published_vectors(native):
    body = (native['worker_frames'][-1].get('publication') or {}).get('body') or {}
    return dict(body.get('page_vectors') or {})


def cosine(x, y):
    dot = sum(a * b for a, b in zip(x, y))
    norm = math.sqrt(sum(a * a for a in x)) * math.sqrt(sum(b * b for b in y))
    return dot / norm if norm else None


def compare_vectors(x, y):
    """Page vectors of the same Pages in two runs (compact_int8_v1 [[q], scale], or legacy float rows): identical
    count, max |dq| (or |dx|), max relative scale gap."""
    def parts(value):
        if compact_vector(value):
            return value[0], value[1]
        return list(value), 1.0
    common = sorted(set(x) & set(y), key=int)
    same, dq, scale = 0, 0, 0.0
    for page in common:
        (qx, sx), (qy, sy) = parts(x[page]), parts(y[page])
        if qx == qy and sx == sy:
            same += 1
        dq = max([dq] + [abs(a - b) for a, b in zip(qx, qy)])
        scale = max(scale, abs(sx - sy) / max(abs(sx), 1e-30))
    return dict(pages=len(common), identical=same, max_abs_dq=dq, max_scale_rel=round(scale, 9),
                missing=sorted(set(x) ^ set(y), key=int)[:10])


def selection_reason(native):
    """A query turn's selection reason ('cold_start', 'below_switch_threshold', None = active), or 'none' without a
    query receipt."""
    selection = (((native.get('query') or {}).get('body') or {}).get('selection')) or {}
    return selection['reason'] if 'reason' in selection else 'none'


def compare_selection(reference, other):
    """observation_query graph run against the eager run of its layout, session by session and turn by turn while
    their prompts agree: the selected Pages, the phase-1 and published int8 Page vectors and the observation query.
    Differences are findings (SELECTION_DIFFERS, PAGE_VECTORS_DIFFER), never failures: compiled kernels change the low
    bits of the hidden rows, so near-tied selections can flip. A pair whose compared turns hold no active selection
    (the replies diverged before the first selecting turn) is the finding ACTIVE_NOT_COMPARED: the selection parity
    is then unanswered, not equal."""
    a, b = run_turns(reference), run_turns(other)
    rows, flipped, vectors_differ, stopped = [], [], [], {}
    for key in sorted(set(a) & set(b)):
        if key[0] in stopped:
            continue
        x, y = a[key], b[key]
        if x['prompt'] != y['prompt']:
            stopped[key[0]] = key[1]
            continue
        sx, sy = list(x['native']['plan']['selected']), list(y['native']['plan']['selected'])
        px, ox = phase1_vectors(x['native'])
        py, oy = phase1_vectors(y['native'])
        reason, reason_other = selection_reason(x['native']), selection_reason(y['native'])
        body = (((x['native'].get('query') or {}).get('body') or {}).get('selection')) or {}
        row = dict(session=key[0], turn=key[1], active=reason is None, reason='active' if reason is None else reason,
                   reason_equal=reason == reason_other, candidates=body.get('candidate_pages'),
                   selected=len(sx), selected_equal=sx == sy,
                   jaccard=round(len(set(sx) & set(sy)) / len(set(sx) | set(sy)), 6) if set(sx) | set(sy) else 1.0,
                   phase1=compare_vectors(px, py), published=compare_vectors(published_vectors(x['native']),
                                                                             published_vectors(y['native'])),
                   observation_cosine=None if ox is None or oy is None else round(cosine(ox, oy), 9))
        rows.append(row)
        if not row['selected_equal']:
            flipped.append([key[0], key[1]])
        if any(v['identical'] != v['pages'] or v['missing'] for v in (row['phase1'], row['published'])):
            vectors_differ.append([key[0], key[1]])
    need(rows, 'no common turn with identical prompts in %s and %s' % (Path(reference).name, Path(other).name))
    active_reference = sorted([list(k) for k, t in a.items() if selection_reason(t['native']) is None])
    compared = [[r['session'], r['turn']] for r in rows if r['active']]
    detail = dict(reference=Path(reference).name, other=Path(other).name, turns=rows, selection_differs=flipped,
                  vectors_differ=vectors_differ, active_turns=len(active_reference), active_reference=active_reference,
                  active_compared=compared, sessions_compared=len({r['session'] for r in rows}),
                  prompts_diverged={str(k): t for k, t in sorted(stopped.items())})
    if flipped:
        raise Finding(SELECTION_FINDING, detail)
    if not compared:
        raise Finding(ACTIVE_COMPARED_FINDING, detail)
    if vectors_differ:
        raise Finding(VECTOR_FINDING, detail)
    return detail


def active_batching(out):
    """Selection under batching (scg2 r3): how many active-selection requests of a multi-session run decoded
    together. At each /metrics batch sample (time t, `running` requests) the requests in flight at t come from the
    clients' send/receive times on the same driver clock; of f requests in flight at most f - running are not
    running, so at least a - (f - running) of the a active-selection requests in flight are running: a lower bound,
    never an estimate. A run whose bound never reached 2 is the finding ACTIVE_NOT_BATCHED (a workload timing, not a
    runtime fault): its graph decode steps then never carried two sessions' selected block tables at once."""
    out = Path(out)
    turns = run_turns(out)
    intervals, k = [], 0
    while (out / ('s%d' % k) / 'session.json').exists():
        for entry in load(out / ('s%d' % k) / 'session.json').get('turns') or ():
            key = (k, entry.get('turn'))
            if key in turns and entry.get('t_send_s') is not None and entry.get('t_recv_s') is not None:
                intervals.append((entry['t_send_s'], entry['t_recv_s'], selection_reason(turns[key]['native']) is None,
                                  key))
        k += 1
    active = sorted([list(i[3]) for i in intervals if i[2]])
    need(active, 'no active-selection turn in %s' % out.name)
    samples, bound_max, together, at = 0, 0, 0, None
    for point in measure.batch_timeline(out):
        if not point['ok'] or point['running'] is None:
            continue
        samples += 1
        t = point['t_s']
        flying = [i for i in intervals if i[0] <= t <= i[1]]
        bound = max(0, sum(1 for i in flying if i[2]) - max(0, len(flying) - int(point['running'])))
        together += bound >= 2
        if bound > bound_max:
            bound_max, at = bound, dict(t_s=t, running=point['running'], in_flight=len(flying),
                                        active_in_flight=sorted(list(i[3]) for i in flying if i[2]))
    overlapping = sorted({tuple(x[3]) for x in intervals if x[2] for y in intervals
                          if y[2] and y[3][0] != x[3][0] and x[0] < y[1] and y[0] < x[1]})
    detail = dict(label=out.name, active_turns=len(active), samples=samples,
                  samples_two_or_more_active_running=together, active_running_lower_bound_max=bound_max, at=at,
                  active_turns_overlapping_another_session=len(overlapping))
    if bound_max < 2:
        raise Finding(ACTIVE_BATCH_FINDING, detail)
    return detail


# ---------------------------------------------------------------- perf-int: job 3339019 as the reference

REFERENCE_TOKENS_FINDING = 'REFERENCE_TOKENS_DIFFER'
REFERENCE_SELECTION_FINDING = 'REFERENCE_SELECTION_DIFFERS'
REFERENCE_ACTIVE_FINDING = 'REFERENCE_ACTIVE_FEWER'
REFERENCE_ADMISSION_FINDING = 'REFERENCE_ADMISSION_SLOWER'


def reference_records(out):
    """{"s<k>-t<t>": record} of one server output directory, in the form of pins.json reference_job (reference_job.py
    seals job 3339019's with this function): prompt tokens and digest, reply tokens and SHA-256, the selection
    (reason, candidates, selected count and digest, query hash), admission_engine_ms and phase-1 ms."""
    out = Path(out)
    rows = {r['response_id']: r for r in measure.runtime_rows(out)}
    records, k = {}, 0
    while (out / ('s%d' % k) / 'session.json').exists():
        client = load(out / ('s%d' % k) / 'session.json')
        for t in range(client['completed_turns']):
            response = load(out / ('s%d' % k) / ('http-response-%d.json' % t))
            row = rows.get(response.get('id'))
            need(row is not None, 's%d turn %d of %s has no runtime row' % (k, t, out.name))
            native = load(native_path(out, row['request_id']))
            content = response['choices'][0]['message'].get('content') or ''
            selected = list(native['plan']['selected'])
            body = ((native.get('query') or {}).get('body')) or {}
            selection = body.get('selection') or {}
            reason = None if not selection else ('active' if selection.get('reason') is None else selection['reason'])
            phase1 = (row['stage_b'].get('phase1') or {}).get('phase1_ns')
            records['s%d-t%d' % (k, t)] = dict(
                prompt_tokens=len(row['prompt_token_ids']), prompt_sha256=row['prompt_token_sha256'],
                completion_tokens=(response.get('usage') or {}).get('completion_tokens'),
                content_sha256=hashlib.sha256(content.encode('utf-8')).hexdigest(), reason=reason,
                candidates=row['candidate_page_count'], selected=len(selected), selected_sha256=digest(selected),
                query_hash=body.get('query_hash'), admission_engine_ms=native.get('admission_engine_ms'),
                phase1_ms=None if phase1 is None else round(phase1 / 1e6, 3))
        k += 1
    return records


def compare_reference(new, old):
    """Turn records of this job's server against job 3339019's (session by session, until the first divergence):
    returns (rows, diverged sessions, selection mismatches on identical prompts)."""
    sessions = sorted({int(key.split('-')[0][1:]) for key in set(new) | set(old)})
    rows, diverged, mismatched = [], [], []
    for k in sessions:
        row = dict(session=k, identical_turns=0, first_divergence=None, selections_compared=0, selections_equal=0,
                   active_new=sum(1 for key, v in new.items() if key.startswith('s%d-' % k) and v['reason'] == 'active'),
                   active_reference=sum(1 for key, v in old.items()
                                        if key.startswith('s%d-' % k) and v['reason'] == 'active'))
        t = 0
        while 's%d-t%d' % (k, t) in new and 's%d-t%d' % (k, t) in old:
            x, y = new['s%d-t%d' % (k, t)], old['s%d-t%d' % (k, t)]
            if x['prompt_sha256'] != y['prompt_sha256']:
                row['first_divergence'] = dict(turn=t, where='prompt', prompt_tokens=[x['prompt_tokens'],
                                                                                      y['prompt_tokens']])
                break
            if x['reason'] is not None or y['reason'] is not None:
                row['selections_compared'] += 1
                same = all(x[f] == y[f] for f in ('reason', 'candidates', 'selected', 'selected_sha256', 'query_hash'))
                row['selections_equal'] += same
                if not same:
                    mismatched.append(dict(session=k, turn=t, **{f: [x[f], y[f]] for f in (
                        'reason', 'candidates', 'selected', 'selected_sha256', 'query_hash') if x[f] != y[f]}))
            if (x['content_sha256'], x['completion_tokens']) != (y['content_sha256'], y['completion_tokens']):
                row['first_divergence'] = dict(turn=t, where='reply', completion_tokens=[x['completion_tokens'],
                                                                                         y['completion_tokens']])
                break
            row['identical_turns'] += 1
            t += 1
        if row['first_divergence'] is not None:
            diverged.append(k)
        rows.append(row)
    return rows, diverged, mismatched


def check_reference_tokens(out, reference, label, old):
    """Tokens of `label` against job 3339019's `old` run, session by session (the same tasks, observations, model and
    tokenizer; ms-pool and cudagraph are byte-identical when off): identical turns until the first difference. A
    difference is the finding REFERENCE_TOKENS_DIFFER, never a failure (the per-server token-stream checks already
    re-render every prompt token-exactly; a reply can change with the batch composition, which depends on timing). A
    prompt that differs at turn 0 is named apart (`turn0_prompt_differs`): it points at the render or the tasks."""
    new = reference_records(out)
    rows, diverged, mismatched = compare_reference(new, reference['runs'][old]['turns'])
    detail = dict(label=label, reference_job=reference['job_id'], reference_run=old, turns=len(new),
                  reference_turns=len(reference['runs'][old]['turns']), rows=rows,
                  identical_sessions=len(rows) - len(diverged),
                  identical_turns=sum(r['identical_turns'] for r in rows), diverged_sessions=diverged,
                  turn0_prompt_differs=[r['session'] for r in rows if (r['first_divergence'] or {}).get('turn') == 0
                                        and r['first_divergence']['where'] == 'prompt'])
    if diverged:
        raise Finding(REFERENCE_TOKENS_FINDING, detail)
    return detail


def check_reference_selection(out, reference, label, old):
    """Selections of `label` against job 3339019's `old` run while the prompts agree (reason, candidates, the selected
    Pages' digest and the observation query hash), and the active-selection turns per session. A selection that
    differs on an identical prompt is the finding REFERENCE_SELECTION_DIFFERS; fewer active turns in a session than
    the reference is REFERENCE_ACTIVE_FEWER. Never failures (the offline selection replay checks the selector)."""
    new = reference_records(out)
    rows, diverged, mismatched = compare_reference(new, reference['runs'][old]['turns'])
    fewer = [dict(session=r['session'], active=r['active_new'], reference=r['active_reference']) for r in rows
             if r['active_new'] < r['active_reference']]
    detail = dict(label=label, reference_run=old, selections_compared=sum(r['selections_compared'] for r in rows),
                  selections_equal=sum(r['selections_equal'] for r in rows), mismatched=mismatched[:20],
                  active_turns=sum(r['active_new'] for r in rows),
                  reference_active_turns=sum(r['active_reference'] for r in rows), fewer_active=fewer)
    if mismatched:
        raise Finding(REFERENCE_SELECTION_FINDING, detail)
    if fewer:
        raise Finding(REFERENCE_ACTIVE_FINDING, detail)
    return detail


def check_reference_metrics(summary, reference, label, old, limit):
    """The measured summary of `label` next to job 3339019's `old` run (pins.json reference_job: admission mean over
    continuing turns, the residual fit, steady ITL and tok/s at the session count, the stall share, decode tok/s):
    a table. An admission mean above (1 + limit) x the reference is the finding REFERENCE_ADMISSION_SLOWER (int-1008
    adds the prompt digest once per binding: the admission should not be slower)."""
    ref = reference['runs'][old]['metrics']
    points = residual_points(summary)
    fit = least_squares(points)
    itl = steady_itl(summary)
    batches = {b['batch']: b for b in ((summary.get('measurements') or {}).get('batch') or {}).get('batches') or ()}
    aggregate = (summary.get('measurements') or {}).get('aggregate') or {}
    turns = [t for t in summary.get('turns') or () if t.get('turn') and type(t.get('admission_engine_ms')) in (int, float)]
    mine = dict(admission_engine_ms_mean=round(sum(t['admission_engine_ms'] for t in turns) / len(turns), 3) if turns else None,
                residual_requests=len(points), residual_slope_us_per_page=None if fit is None else round(fit[0] * 1000, 3),
                residual_intercept_ms=None if fit is None else round(fit[1], 3),
                steady_itl_ms_at_sessions=(itl.get(summary.get('sessions')) or (None,))[0],
                steady_tok_s_at_sessions=(batches.get(summary.get('sessions')) or {}).get('steady_tok_s'),
                admission_stall_fraction=aggregate.get('admission_stall_fraction'),
                gen_tokens_per_s=aggregate.get('gen_tokens_per_s'),
                active_selection_turns=summary.get('active_selection_turns'))
    table = [dict(metric=k, value=v, reference=ref.get(k),
                  ratio=round(v / ref[k], 4) if type(v) in (int, float) and type(ref.get(k)) in (int, float) and ref[k]
                  else None) for k, v in mine.items()]
    detail = dict(label=label, reference_job=reference['job_id'], reference_run=old, limit=limit, table=table)
    base, value = ref.get('admission_engine_ms_mean'), mine['admission_engine_ms_mean']
    if base and value and value > (1 + limit) * base:
        raise Finding(REFERENCE_ADMISSION_FINDING, detail)
    return detail


def component_rows(summary):
    """Per valid continuing turn of a run: the admission (engine thread) and the parts the evidence names: phase 1
    (stage_b phase1_ns), the worker commands of the admission (prepare, phase1, phase1_drain, reserve: their worker-side
    elapsed time) and the rest."""
    out = []
    for row in ((summary.get('measurements') or {}).get('rows') or ()):
        if not (row.get('valid') and row.get('turn')):
            continue
        stall, phase1 = row.get('admission_engine_ms'), row.get('phase1_ms')
        ops = row.get('worker_op_ms') or {}
        if None in (stall, phase1):
            continue
        worker = {name: (ops.get(name) or {}).get('ms') for name in ('prepare', 'phase1', 'phase1_drain', 'reserve')}
        own = admission_phase1_ms(row)
        others = sum(v for k, v in worker.items() if k != 'phase1' and v is not None)
        out.append(dict(pages=(row.get('pages') or {}).get('candidates'), admission_ms=stall, phase1_ms=phase1,
                        residual_ms=round(stall - own, 3), worker_prepare_ms=worker['prepare'],
                        worker_phase1_ms=worker['phase1'], worker_phase1_drain_ms=worker['phase1_drain'],
                        worker_reserve_ms=worker['reserve'], phase1_parked=bool(row.get('phase1_parked')),
                        residual_outside_worker_ms=round(stall - own - others, 3)))
    return out


def check_components(by_label, reference=None, pairs=()):
    """The admission residual's components (perf-int: what the evidence exposes, no runtime change): per run, the mean
    and median of admission_engine_ms, phase-1 ms, the residual (admission - phase 1), the worker-side time of the
    admission's other worker commands (prepare, phase1_drain, reserve) and the residual outside them (the engine
    thread's own work: render checks, capacity, selection, plan, RPC); with `reference` the same means of job
    3339019's runs (admission and phase 1 only: its records). Informational: never a finding."""
    def stats(values):
        values = sorted(v for v in values if type(v) in (int, float))
        if not values:
            return None
        return dict(mean=round(sum(values) / len(values), 3), p50=values[len(values) // 2], n=len(values))
    table = []
    for label, summary in by_label.items():
        rows = component_rows(summary)
        table.append(dict(label=label, source='this job', requests=len(rows), **{
            key: stats([r[key] for r in rows]) for key in ('admission_ms', 'phase1_ms', 'residual_ms', 'worker_prepare_ms',
                                                           'worker_phase1_ms', 'worker_phase1_drain_ms',
                                                           'worker_reserve_ms', 'residual_outside_worker_ms')}))
    for label, old in pairs:
        turns = [v for key, v in ((reference or {}).get('runs', {}).get(old, {}).get('turns') or {}).items()
                 if not key.endswith('-t0') and type(v.get('admission_engine_ms')) in (int, float)
                 and type(v.get('phase1_ms')) in (int, float)]
        table.append(dict(label=old, source='job %s (reference of %s)' % ((reference or {}).get('job_id'), label),
                          requests=len(turns), admission_ms=stats([v['admission_engine_ms'] for v in turns]),
                          phase1_ms=stats([v['phase1_ms'] for v in turns]),
                          residual_ms=stats([round(v['admission_engine_ms'] - v['phase1_ms'], 3) for v in turns])))
    return dict(table=table)


# ---------------------------------------------------------------- perf-int: the session pool (ms-pool, stage 2)

SESSION_EVENT_SCHEMA = 'remote-ra-session-event-v1'
CLOSE_OBJECT = 'session.close'
WORKER_RELEASE_KEYS = {'catalog_pages', 'carry', 'terminals', 'phase1_leases', 'arena'}


def pool_of(expect):
    """K of a session-pool server's expectation (`running`), or None for a fixed-session server."""
    return expect.get('running')


def session_events(out):
    """sessions.jsonl rows (open and close events of a session-pool server, in engine order)."""
    path = Path(out) / 'sessions.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def check_session_pool(out, expect, rows, clients, ids, ready, timeline, tolerance, policy=None):
    """The session pool (ms-pool-report "What the perf harnesses need"; runtime session_pool.py) as the evidence shows it.

    - Registration: READY's session_pool is {max_running_sessions K, registered T}, startup.json pins
      max_running_sessions K, READY lists all T sessions and the worker every registered session id.
    - Lazy open: sessions.jsonl holds exactly one open row per task that sent a request, none for a task that never did,
      every open row's open_sessions <= K, the j-th open (j >= K) after at least j - K + 1 closes (never more than K
      open).
    - FIFO open wait and refill order: the engine opens the tasks in the order their turn-0 requests were sent (the
      driver starts them in task order; two sends closer than ORDER_TOLERANCE_SECONDS may swap), so a later request
      never opens before an earlier waiter; every turn-0 row carries open_wait_s (later turns 0); a task that waited
      (> 0) opened (client send + open_wait_s) inside one of the harness's close calls (within the tolerance): the slot
      passed on at a close; with a window > K some task waited (the FIFO was exercised). `opened_in_task_order` records
      whether the arrival order was the registration order.
    - Close via the loopback route: every task that ran was closed by the driver with HTTP 200, the receipt (minus
      `object` = session.close) equals its sessions.jsonl close row, event close, reason close, opened, turns = the
      task's completed turns, its instance id and index; a repeated close returns the same receipt with `repeated` true;
      an unregistered id is refused with 404 SESSION_UNKNOWN.
    - Each close releases exactly the session's pins: released.blocks = the FA pool's free blocks after - before =
      resident Pages + the live tail, the Mamba pool unchanged (idle sessions pin no GDN slot), released.requests =
      the turns, the worker's catalog Pages, carry, terminal receipts (= turns) and phase-1 leases (= turns - 1) and its
      arena stats; the per-session selection tables and Goal state dropped (goal_state inline / goal_thread on a
      selective server, None on full). A close that leaves no session open puts both pools back at READY's baseline.
    - At shutdown: engine-cleanup.json's session_pool names every task that ran as closed (none open at shutdown, none
      closed unopened), the never-started tasks as never_opened; the parent receipt names all T sessions.
    - Request bound: the admitted requests (every session's turns) stay within max(1,024, 128 x T), and all T sessions
      admitted over the server's lifetime (the session table is sized T, not K)."""
    out = Path(out)
    K, T = pool_of(expect), len(expect['tasks'])
    startup = load(out / 'startup.json')
    need(ready.get('session_pool') == dict(max_running_sessions=K, registered=T),
         'READY session_pool %r, expected K = %d of T = %d' % (ready.get('session_pool'), K, T))
    need(startup.get('max_running_sessions') == K, 'startup max_running_sessions %r' % startup.get('max_running_sessions'))
    events = session_events(out)
    need(events, 'sessions.jsonl absent or empty')
    for e in events:
        need(e.get('schema') == SESSION_EVENT_SCHEMA and e.get('event') in ('open', 'close')
             and e.get('max_running_sessions') == K and type(e.get('session_index')) is int
             and 0 <= e['session_index'] < T and e.get('session_id') == ids[e['session_index']]
             and e.get('instance_id') == expect['tasks'][e['session_index']]['instance_id']
             and type(e.get('monotonic_ns')) is int, 'malformed sessions.jsonl row %r' % ({k: e.get(k) for k in (
                 'schema', 'event', 'session_index', 'session_id', 'instance_id', 'max_running_sessions')},))
    need([e['monotonic_ns'] for e in events] == sorted(e['monotonic_ns'] for e in events),
         'sessions.jsonl rows are not in time order')
    ran = [k for k, c in enumerate(clients) if c.get('completed_turns') or c.get('turns')]
    opens = [e for e in events if e['event'] == 'open']
    closes = [e for e in events if e['event'] == 'close']
    need(sorted(e['session_index'] for e in opens) == ran,
         'open rows for tasks %s, tasks that sent a request %s' % (sorted(e['session_index'] for e in opens), ran))
    order = [e['session_index'] for e in opens]
    send0 = {k: clients[k]['turns'][0]['t_send_s'] for k in ran if clients[k].get('turns')}
    late = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]
            if a in send0 and b in send0 and send0[b] + ORDER_TOLERANCE_SECONDS < send0[a]]
    need(not late, 'sessions opened out of their turn-0 arrival order (FIFO): (opened first, sent first) %s; opens %s'
         % ([(a, b, send0[a], send0[b]) for a, b in late[:4]], order))
    need(all(1 <= e['open_sessions'] <= K for e in opens), 'an open row with more than %d sessions open: %s'
         % (K, [e['open_sessions'] for e in opens]))
    seen_closes, open_now, peak = 0, 0, 0
    for e in events:
        if e['event'] == 'open':
            open_now += 1
            index = opens.index(e)
            need(index < K or seen_closes >= index - K + 1, 'open %d of task %d came after only %d closes (K = %d)'
                 % (index, e['session_index'], seen_closes, K))
        else:
            seen_closes += 1
            open_now -= 1 if e.get('opened') else 0
        peak = max(peak, open_now)
        need(open_now <= K and e['open_sessions'] == open_now, 'open sessions %r at %s of task %d, counted %d'
             % (e['open_sessions'], e['event'], e['session_index'], open_now))
    # FIFO open wait, on the client clock: the open lies inside one of the driver's close calls.
    driver = load(out / 'driver.json')
    calls = sorted(driver.get('closes') or (), key=lambda c: c['t_start_s'])
    window = driver.get('window')
    by = {(r['session_index'], r['turn']): r for r in rows}
    waits = []
    for k in ran:
        row0 = by.get((k, 0))
        need(row0 is not None, 's%d turn 0 has no runtime row' % k)
        for t in range(clients[k]['completed_turns']):
            wait = by[(k, t)].get('open_wait_s')
            need(type(wait) in (int, float) and math.isfinite(wait) and wait >= 0,
                 's%d turn %d: open_wait_s missing or invalid: %r' % (k, t, wait))
            need(t == 0 or wait == 0, 's%d turn %d waited %.3f s to open after turn 0' % (k, t, wait))
        wait = row0['open_wait_s']
        send = clients[k]['turns'][0]['t_send_s']
        if wait > 0:  # it waited for a slot: a close passed it on
            at = send + wait
            hit = [c for c in calls if c['t_start_s'] - tolerance <= at <= c['t_end_s'] + tolerance]
            need(hit, 's%d opened at %.3f s (send %.3f + wait %.3f), outside every close call %s'
                 % (k, at, send, wait, [(c['session'], c['t_start_s'], c['t_end_s']) for c in calls][:6]))
        waits.append(dict(session=k, open_wait_s=round(wait, 3)))
    need(not (window and window > K and len(ran) > K) or any(w['open_wait_s'] > 0 for w in waits),
         'no task waited in the open FIFO (window %r > K = %d, %d tasks ran): the FIFO open wait was not exercised'
         % (window, K, len(ran)))
    # The loopback close: one 200 per task that ran, its receipt equal to its close row.
    receipts = {}
    for k in ran:
        call = next((c for c in calls if c['session'] == k), None)
        need(call is not None and call['status'] == 200, 's%d: the driver did not close it (%r)' % (k, call))
        receipt = load(out / ('s%d' % k) / 'close.json')
        row = next((e for e in closes if e['session_index'] == k), None)
        need(row is not None and receipt.get('object') == CLOSE_OBJECT
             and {key: v for key, v in receipt.items() if key != 'object'} == row,
             's%d: close receipt %r differs from its sessions.jsonl row' % (k, {key: receipt.get(key) for key in (
                 'object', 'event', 'opened')}))
        need(row['reason'] == 'close' and row['opened'] is True and row['turns'] == clients[k]['completed_turns']
             and row.get('rebuilds') == 0 and row.get('poisoned') is False,
             's%d: close row reason %r opened %r turns %r (completed %d) rebuilds %r poisoned %r'
             % (k, row.get('reason'), row.get('opened'), row.get('turns'), clients[k]['completed_turns'],
                row.get('rebuilds'), row.get('poisoned')))
        receipts[k] = row
    need(len(closes) == len(ran), '%d close rows for %d tasks that ran' % (len(closes), len(ran)))
    # Exactly the session's pins come back; the Mamba pool does not move; the selection tables go.
    baseline = list(ready.get('free_blocks_by_pool') or ())
    selective = policy in SELECTIVE
    released = []
    order = [e['session_index'] for e in closes if e['session_index'] in receipts]  # close order (sessions.jsonl)
    need(not order or receipts[order[-1]]['open_sessions'] == 0,
         'the last close leaves %r open sessions' % (receipts[order[-1]]['open_sessions'] if order else None))
    for k in order:
        row = receipts[k]
        rel, before, after = row.get('released') or {}, row.get('free_blocks_by_pool_before'), row.get('free_blocks_by_pool')
        need(type(before) is list and type(after) is list and len(before) == len(after) == 2,
             's%d: close pool counts %r / %r' % (k, before, after))
        need(rel.get('blocks') == after[0] - before[0] and after[1] == before[1],
             's%d: the close released %r blocks, the FA pool moved %d, the Mamba pool %d'
             % (k, rel.get('blocks'), after[0] - before[0], after[1] - before[1]))
        need(rel.get('blocks') == rel.get('resident_pages') + (1 if rel.get('tail') else 0),
             's%d: released blocks %r != resident Pages %r + tail %r' % (k, rel.get('blocks'), rel.get('resident_pages'),
                                                                          rel.get('tail')))
        turns = row['turns']
        worker = rel.get('worker') or {}
        need(rel.get('requests') == turns and set(worker) == WORKER_RELEASE_KEYS and worker['carry'] is True
             and worker['terminals'] == turns and worker['phase1_leases'] == turns - 1
             and type(worker['catalog_pages']) is int and worker['catalog_pages'] > 0
             and isinstance(worker['arena'], dict),
             's%d: released requests %r, worker %r (turns %d)' % (k, rel.get('requests'), worker, turns))
        need(row.get('goal_state') in (('inline', 'goal_thread') if selective else (None,)),
             's%d: goal_state %r on a %s server (the selection tables must be dropped)'
             % (k, row.get('goal_state'), policy))
        if row['open_sessions'] == 0:
            need(after == baseline, 's%d: the last open session closed and the pools are %r, READY baseline %r'
                 % (k, after, baseline))
        released.append(dict(session=k, blocks=rel['blocks'], resident_pages=rel['resident_pages'],
                             catalog_pages=worker['catalog_pages'], arena_pages_held=worker['arena'].get('pages_held'),
                             open_after=row['open_sessions']))
    # Repeated and unknown closes.
    extra = {e['kind']: e for e in driver.get('extra_closes') or ()}
    if receipts and not driver.get('deadline_stop_s'):
        repeated, unknown = extra.get('repeated') or {}, extra.get('unknown') or {}
        first = repeated.get('session')
        body = repeated.get('body') or {}
        need(repeated.get('status') == 200 and body.get('repeated') is True and first in receipts
             and {key: v for key, v in body.items() if key not in ('object', 'repeated')} == receipts[first],
             'a repeated close returned %r %r' % (repeated.get('status'), {key: body.get(key) for key in ('repeated',
                                                                                                          'event')}))
        need(unknown.get('status') == 404 and 'SESSION_UNKNOWN' in json.dumps(unknown.get('body')),
             'a close of an unregistered id returned %r %r' % (unknown.get('status'), unknown.get('body')))
    # Shutdown and the bounds.
    cleanup = load(out / 'engine-cleanup.json')
    never = [ids[k] for k in range(T) if k not in ran]
    need(cleanup.get('session_pool') == dict(max_running_sessions=K, registered=T, closed=[ids[k] for k in ran],
                                             closed_unopened=[], open_at_shutdown=[], never_opened=never),
         'engine-cleanup session_pool %r' % (cleanup.get('session_pool'),))
    bound = max(1024, 128 * T)
    need(len(rows) <= bound and len(cleanup.get('admitted_request_ids') or ()) == len(rows),
         '%d admitted requests, bound %d' % (len(rows), bound))
    arena = [r['arena_pages_held'] for r in released if type(r.get('arena_pages_held')) is int]
    return dict(running=K, registered=T, ran=len(ran), opens=len(opens), closes=len(closes), open_peak=peak, waits=waits,
                open_order=order, opened_in_task_order=order == sorted(order),
                waited=len([w for w in waits if w['open_wait_s'] > 0]), open_wait_s_max=max(
                    (w['open_wait_s'] for w in waits), default=0.0), request_bound=bound, requests=len(rows),
                released=released, arena_pages_held_max=max(arena, default=None),
                close_attempts_max=max((len(c['attempts']) for c in calls), default=0), never_opened=never)


# ---------------------------------------------------------------- perf-ctx: the context limit and the reply bound (align-ctx)

DEFAULT_MAX_MODEL_LEN = 65536
CONTEXT_RECORD_KEY = 'context'  # remote_ra_live_v2.context_limits.RECORD_KEY: ready.json and the worker evidence
LONG_REPLY_FINDING = 'LONG_REPLY_NOT_EXERCISED'
HOST_PIN_FINDING = 'HOST_PIN_ABOVE_PREDICTED'
HOST_PIN_HIDDEN = 'HOST_PIN_NOT_VISIBLE'
HOST_PIN_SLACK_BYTES = 2 * 1024 ** 3
# What an over-length refusal names (vLLM's VLLMValidationError "This model's maximum context length is ... you
# requested 4096 output tokens ...", or the engine's check_sampling "... differs: max_tokens=N").
OVERLENGTH_REFUSAL = re.compile(r'maximum context length|max_tokens|max_model_len|context length', re.I)
GIB_BYTES = 1024 ** 3
CONNECTOR_BYTES = 32 * GIB_BYTES
ARENA_CHUNK_BYTES = 256 << 20
MAX_SEQ_LEN = re.compile(r'\bmax_seq_len=(\d+)')


def context_of(expect):
    """(max_model_len, max_tokens) a server's expectation pins (the defaults when it pins none)."""
    return expect.get('max_model_len') or DEFAULT_MAX_MODEL_LEN, expect.get('max_tokens') or MAX_TOKENS


def check_context_limits(out, expect, server_log, turns):
    """align-ctx's startup record and bounds, as the server's evidence shows them (checked once at startup by the
    runtime; here once per server): the effective config's max_model_len (vLLM's engine config line `max_seq_len`
    when the log has it) is the pinned context limit; startup.json pins `max_tokens` only when it is not 1536 and
    `dram_arena_pages` only with --dram-arena-gib; ready.json and the worker evidence (worker-attached.json receipt)
    carry the same `context` record {max_model_len, max_tokens} when either value is not the default and none
    otherwise; a decoupled server's pinned arena (receipt `dram_arena`) has the planned capacity in Pages of P x 65,536
    bytes; and every completed turn keeps prompt + completion within the context limit and its completion within the
    reply bound."""
    out = Path(out)
    L, T = context_of(expect)
    ready, startup = load(out / 'ready.json'), load(out / 'startup.json')
    receipt = load(out / 'worker-attached.json')['receipt']
    effective = ready.get('effective_server_config') or {}
    need(effective.get('max_model_len') == L and (startup.get('effective_server_config') or {}).get('max_model_len') == L,
         'effective max_model_len %r (startup %r), expected %d' % (effective.get('max_model_len'),
                                                                  (startup.get('effective_server_config') or {})
                                                                  .get('max_model_len'), L))
    need(('max_tokens' in startup) == (T != MAX_TOKENS) and startup.get('max_tokens', MAX_TOKENS) == T,
         'startup max_tokens %r, expected %s' % (startup.get('max_tokens'), 'no key' if T == MAX_TOKENS else T))
    record = None if (L, T) == (DEFAULT_MAX_MODEL_LEN, MAX_TOKENS) else dict(max_model_len=L, max_tokens=T)
    need((CONTEXT_RECORD_KEY in ready) == (record is not None) and ready.get(CONTEXT_RECORD_KEY) == record,
         'ready.json context %r, expected %r' % (ready.get(CONTEXT_RECORD_KEY), record))
    need((CONTEXT_RECORD_KEY in receipt) == (record is not None) and receipt.get(CONTEXT_RECORD_KEY) == record,
         'the worker echoes context %r, expected %r' % (receipt.get(CONTEXT_RECORD_KEY), record))
    layout = layout_of(expect)
    pinned = layout.get('dram_arena_pages')
    need(('dram_arena_pages' in startup) == (pinned is not None) and startup.get('dram_arena_pages') == pinned,
         'startup dram_arena_pages %r, expected %r' % (startup.get('dram_arena_pages'), pinned))
    arena = receipt.get('dram_arena')
    if layout.get('decoupled') and layout.get('arena_pages') is not None:
        # remote_ra_growing_worker.tensors.PageArena: whole 256-MiB chunks, so the capacity is the planned Pages
        # rounded up to a chunk (8,193 -> 8,448 at P = 16; 90,112 stays).
        page_bytes = layout.get('page_bytes') or layout['page_tokens'] * FA_TOKEN_BYTES  # (Gemma: P x 81,920)
        chunk = max(1, ARENA_CHUNK_BYTES // page_bytes)
        capacity = -(-layout['arena_pages'] // chunk) * chunk
        need(isinstance(arena, dict) and arena.get('max_pages') == capacity
             and arena.get('page_bytes') == page_bytes and arena.get('chunk_slots') == chunk,
             'worker dram_arena %r, expected max_pages %d (%d planned) of %d bytes in chunks of %d'
             % (arena, capacity, layout['arena_pages'], page_bytes, chunk))
    # the main engine's config line (a vLLM Goal rewriter child's own max_seq_len is information: engine_logs)
    engines = engine_logs(out, Path(server_log).read_text(encoding='utf-8', errors='replace'))
    seen = sorted({int(v) for v in MAX_SEQ_LEN.findall(engines['main'])})
    need(not seen or seen == [L], "vLLM's engine config line names max_seq_len %s, expected %d" % (seen, L))
    rows, over = [], []
    for turn in turns:
        prompt, completion = turn.get('prompt_tokens'), (turn.get('http') or {}).get('completion_tokens')
        if type(prompt) is int and type(completion) is int:
            rows.append((prompt, completion))
            if prompt + completion > L or completion > T:
                over.append(dict(session=turn['session'], turn=turn['turn'], prompt=prompt, completion=completion))
    need(not over, 'turns beyond the context limit %d or the reply bound %d: %s' % (L, T, over[:4]))
    return dict(max_model_len=L, max_tokens=T, context=record, dram_arena_pages=pinned,
                arena={k: (arena or {}).get(k) for k in ('max_pages', 'page_bytes', 'chunk_slots', 'chunks', 'pinned_bytes',
                                                          'pages_held', 'pinned')} if arena else None,
                engine_max_seq_len=seen[0] if seen else None, turns=len(rows),
                prompt_tokens_max=max((p for p, _ in rows), default=None),
                completion_tokens_max=max((c for _, c in rows), default=None),
                total_tokens_max=max((p + c for p, c in rows), default=None),
                **(dict(rewriter_engines=rewriter_engine_info(engines)) if engines['split'] else {}))


def check_long_reply(out, expect, clients, turns):
    """A long-context session's long-reply turn (session.ContextPlan): the session sent exactly one, as its last turn,
    with at least session.LONG_REPLY_MIN_LINES lines and a prompt within the context limit minus the reply bound; its
    reply stayed within the reply bound. A reply of at most 1,536 tokens (the model stopped early: the 4096 bound was not
    exercised) is the finding LONG_REPLY_NOT_EXERCISED, a partial pass; so is a session the time budget stopped before
    its long-reply turn."""
    L, T = context_of(expect)
    client = clients[0] if clients else {}
    plan = client.get('plan') or []
    sent = [p for p in plan if p.get('kind') == 'long_reply']
    if not sent and client.get('stop_reason') in ('skipped_time_budget', 'skipped_context_budget'):
        raise Finding(LONG_REPLY_FINDING, dict(reason='the session stopped before its long-reply turn',
                                               stop_reason=client.get('stop_reason'), planned=plan))
    need(len(sent) == 1, 'the long-context session planned %d long-reply turns (plan %s)' % (len(sent), plan))
    turn = sent[0]['turn']
    need(client.get('completed_turns') == turn + 1 and client.get('planned_turns') == turn + 1,
         'the long-reply turn %d is not the session\'s last completed turn (completed %r, planned %r)'
         % (turn, client.get('completed_turns'), client.get('planned_turns')))
    need(sent[0].get('lines', 0) >= 300 and sent[0].get('prompt_high', L) <= L - T,
         'the long-reply turn asked for %r lines with a prompt of up to %r tokens (limit %d)'
         % (sent[0].get('lines'), sent[0].get('prompt_high'), L - T))
    row = next((x for x in turns if x['session'] == 0 and x['turn'] == turn), None)
    need(row is not None, 'the long-reply turn %d has no checked turn' % turn)
    completion = (row.get('http') or {}).get('completion_tokens')
    need(type(completion) is int and completion <= T and row['prompt_tokens'] <= L - T,
         'the long-reply turn: prompt %r, completion %r (bound %d, limit %d)' % (row['prompt_tokens'], completion, T, L))
    detail = dict(turn=turn, lines=sent[0].get('lines'), prompt_tokens=row['prompt_tokens'], completion_tokens=completion,
                  stop=reply_stop(row, T), hit_reply_bound=completion == T,
                  prompt_estimate=[sent[0].get('prompt_low'), sent[0].get('prompt_high')],
                  client_latency_s=row.get('client_latency_s'), growth_turns=sum(1 for p in plan if p.get('kind') == 'growth'),
                  tokens_per_line=client.get('tokens_per_line'))
    if completion <= MAX_TOKENS:
        raise Finding(LONG_REPLY_FINDING, dict(detail, reason='the reply ended within %d tokens' % MAX_TOKENS))
    return detail


def overlength_record(out, clients, expect):
    """The over-length probe of a long-context session and its failures.jsonl rows (session 0, the turn after the last
    completed one): (probe record or None, [failure rows], {request_sha256 of those rows' internal ids})."""
    if not expect.get('context') or not clients:
        return None, [], set()
    value = clients[0].get('overlength')
    if value is None:
        return None, [], set()
    rows = [f for f in failure_rows(out) if f.get('session_index') == 0 and f.get('turn') == value.get('turn')]
    keys = {hashlib.sha256(f['request_id'].encode()).hexdigest() for f in rows if isinstance(f.get('request_id'), str)}
    return value, rows, keys


def check_overlength(out, expect, clients, tokens, server_log, natives_last):
    """The over-length probe (session.overlength_probe), sent after the long-reply turn: a request whose prompt lies
    above the context limit minus the reply bound and within the context limit must be refused cleanly: a non-200
    answer within its timeout; the server answering GET /metrics afterwards; no runtime row for it (evidence-join);
    at most one failures.jsonl row for it, naming the context limit or the reply bound (vLLM's "maximum context length",
    or the engine's "max_tokens"); in the server log either no admission for it (refused in the frontend, before the
    engine) or exactly one, rejected for the same reason (refused by the engine before allocation); the pools back at
    the baseline (kv-pools) and the cleanup otherwise clean (cleanup, which leaves this row out). Its prompt is
    re-derived token-exactly from the last turn's prompt, reply, boundary, the probe's user message and the header (an
    exact tokenizer), else from the driver's bounds."""
    L, T = context_of(expect)
    value, rows, keys = overlength_record(out, clients, expect)
    client = clients[0] if clients else {}
    if value is None:
        need(client.get('stop_reason') in ('skipped_time_budget', 'skipped_context_budget')
             or not any(p.get('kind') == 'long_reply' for p in client.get('plan') or ()),
             'the long-context session sent no over-length probe after its long-reply turn')
        raise Finding(LONG_REPLY_FINDING, dict(reason='no over-length probe (the session stopped before it)',
                                               stop_reason=client.get('stop_reason')))
    status = value.get('status')
    need(status is not None and status != 200, 'the over-length probe was answered with %r (%s)'
         % (status, value.get('error') or value.get('body', '')[:300]))
    need(value.get('health_status') == 200, 'GET /metrics after the probe answered %r (%s)'
         % (value.get('health_status'), value.get('health_error')))
    out = Path(out)
    request = load(out / 's0' / 'overlength-request.json')
    exact = None
    if tokens.exact and natives_last is not None:
        prompt, completion, content = natives_last
        user = '<|im_start|>user\n' + request['messages'][-1]['content'] + IM_END_NL
        boundary = IM_END_NL if (MARKER in content or completion == T) else '\n'
        exact = len(prompt) + completion + len(tokens.encode(boundary)) + len(tokens.encode(user)) + \
            len(tokens.encode(HEADER))
        need(L - T < exact <= L, 'the probe\'s prompt is %d tokens, outside (%d, %d]' % (exact, L - T, L))
    else:
        need(L - T < value['prompt_low'] and value['prompt_high'] <= L, 'the probe\'s prompt bounds %r-%r lie outside '
             '(%d, %d]' % (value['prompt_low'], value['prompt_high'], L - T, L))
    need(len(rows) <= 1, '%d failures.jsonl rows for the probe' % len(rows))
    text = json.dumps(rows, sort_keys=True) + ' ' + (value.get('body') or '')
    need(OVERLENGTH_REFUSAL.search(text) is not None,
         'the refusal names neither the context limit nor the reply bound: %s' % text[:600])
    found = measure.admissions(Path(server_log).read_text(encoding='utf-8', errors='replace'))
    entries = [v for key in keys for v in found['requests'].get(key, [])]
    rejected = [v for v in entries if v.get('rejected') is not None]
    need(len(entries) == len(rejected) <= 1 and all(OVERLENGTH_REFUSAL.search(str(v['rejected'])) for v in rejected),
         'the probe\'s admissions %s' % [(v.get('outcome'), v.get('rejected')) for v in entries])
    return dict(status=status, where='engine' if rejected else 'frontend', prompt_tokens=exact,
                prompt_bounds=[value.get('prompt_low'), value.get('prompt_high')], refused_above=L - T, max_model_len=L,
                lines=value.get('lines'), latency_s=value.get('latency_s'), failure_rows=len(rows),
                health_status=value.get('health_status'),
                refusal=(rows[0].get('detail') if rows else (value.get('body') or ''))[:400])


def check_capture_sizes(out, expect, server_log):
    """A CUDA-graph server's capture sizes are the pinned ones (pins.segments.<s>.capture_sizes; vLLM 0.24.0 at 16
    sequences: 1, 2, 4, 8, 16, 24, 32): the worker record and the engine config line list exactly them, the PIECEWISE
    capture (piecewise, full) took every size and the FULL decode capture (full, full-decode) every size up to the
    server's sequences. The main engine's lines only: a vLLM Goal rewriter child's own sizes and captures are
    information (engine_logs, `rewriter_engines`)."""
    out = Path(out)
    sizes, mode = list(expect['capture_sizes']), layout_of(expect).get('cuda_graphs', 'off')
    record = load(out / 'worker-attached.json')['receipt'].get('cuda_graphs') or {}
    engines = engine_logs(out, Path(server_log).read_text(encoding='utf-8', errors='replace'))
    text = engines['main']
    config = engine_config(text)
    need(record.get('capture_sizes') == sizes and config['cudagraph_capture_sizes'] == sizes,
         'capture sizes: worker %r, engine config line %r, pinned %r' % (record.get('capture_sizes'),
                                                                        config['cudagraph_capture_sizes'], sizes))
    seqs = expect.get('max_num_seqs') or len(expect['tasks'])
    counts = capture_counts(text)
    wanted = {}
    if mode in ('piecewise', 'full'):
        wanted['mixed prefill-decode, PIECEWISE'] = (len(sizes), len(sizes))
    if mode in ('full', 'full-decode'):
        full = len([s for s in sizes if s <= seqs])
        wanted['decode, FULL'] = (full, full)
    need(all(counts.get(k) == v for k, v in wanted.items()), 'captures %s, expected %s' % (counts, wanted))
    return dict(sizes=sizes, mode=mode, sessions=seqs, captures={k: list(v) for k, v in counts.items()},
                **(dict(rewriter_engines=rewriter_engine_info(engines)) if engines['split'] else {}))


def last_turn_tokens(out, rows, k=0):
    """(prompt token ids, completion tokens, reply content) of session k's last completed turn, or None."""
    out = Path(out)
    client = load(out / ('s%d' % k) / 'session.json') if (out / ('s%d' % k) / 'session.json').exists() else {}
    t = (client.get('completed_turns') or 0) - 1
    if t < 0:
        return None
    response = load(out / ('s%d' % k) / ('http-response-%d.json' % t))
    row = next((r for r in rows if r.get('response_id') == response.get('id')), None)
    if row is None:
        return None
    content = response['choices'][0]['message'].get('content') or ''
    return row['prompt_token_ids'], (response.get('usage') or {}).get('completion_tokens'), content


def reply_stop(turn, max_tokens):
    """How a turn's reply ended, from its HTTP check (the next turn's token-stream check names it only when a next turn
    exists): the marker, the reply bound (`length`) or EOS."""
    http = turn.get('http') or {}
    if not http:
        return None
    return 'marker' if http.get('marker') else 'length' if http.get('completion_tokens') == max_tokens else 'eos'


def long_context_rows(summary):
    """Per completed turn of a run: its kind in the long-context plan, prompt and reply tokens, the reply's time per
    output token (vLLM's per-request histogram), admission and phase 1, the restored and the phase-1 H2D bytes, the
    candidate and selected Pages and the active KV (the measurement rows and the turn checks)."""
    measured = {(r.get('session'), r.get('turn')): r for r in ((summary.get('measurements') or {}).get('rows') or ())}
    kinds = {}
    for p in summary.get('context_plan') or ():
        kinds[p['turn']] = p['kind']
    limits = next((c.get('detail') for c in summary.get('checks') or () if c.get('check') == 'context-limits'), None)
    bound = (limits if isinstance(limits, dict) else {}).get('max_tokens') or MAX_TOKENS
    out = []
    for turn in summary.get('turns') or ():
        m = measured.get((turn['session'], turn['turn'])) or {}
        out.append(dict(label=summary.get('label'), policy=summary.get('policy'), session=turn['session'],
                        turn=turn['turn'], kind=kinds.get(turn['turn'], 'initial' if turn['turn'] == 0 else 'turn'),
                        prompt_tokens=turn.get('prompt_tokens'),
                        completion_tokens=(turn.get('http') or {}).get('completion_tokens'),
                        stop=reply_stop(turn, bound), tpot_ms=m.get('tpot_ms'),
                        admission_engine_ms=turn.get('admission_engine_ms'), phase1_ms=m.get('phase1_ms'),
                        phase1_tokens=(turn.get('stage_b') or {}).get('phase1_tokens'),
                        restore_h2d_bytes=turn.get('restore_h2d_bytes'), prepare_h2d_bytes=turn.get('prepare_h2d_bytes'),
                        candidate_pages=(turn.get('selection') or {}).get('candidate_pages'),
                        selected_pages=(turn.get('selection') or {}).get('selected'),
                        active=(turn.get('selection') or {}).get('active') is True,
                        active_kv_bytes=turn.get('active_kv_bytes'), client_latency_s=turn.get('client_latency_s'),
                        server_latency_s=turn.get('server_latency_s')))
    return out


def check_long_context(by_label):
    """Run A's numbers at long context (a table, never a failure): per run every turn's prompt, reply, time per output
    token, admission and phase 1, restore bytes and selection (long_context_rows), the long-reply turn's decode, and the
    over-length probe's outcome (the per-server checks long-reply and overlength-refused decide)."""
    table, runs = [], {}
    for label, summary in by_label.items():
        rows = long_context_rows(summary)
        table.extend(rows)
        detail = {c['check']: c.get('detail') for c in summary.get('checks') or ()
                  if c['check'] in ('long-reply', 'overlength-refused', 'context-limits')}
        runs[label] = dict(long_reply=detail.get('long-reply'), overlength=detail.get('overlength-refused'),
                           context=detail.get('context-limits'),
                           prompt_tokens_max=max((r['prompt_tokens'] or 0 for r in rows), default=None))
    return dict(table=table, runs=runs)


def check_host_memory(by_label, pins, options):
    """Host memory of every run (perf-ctx item 3; resources.py: the server group's summed VmPin / VmRSS / VmHWM and the
    job cgroup's memory.current / memory.peak every second) next to the launcher's model of pinned host memory: the
    32 GiB connector pool, one Gated-DeltaNet carry (153,944,064 B) per running session and the pinned arena's
    256-MiB chunks for the Pages the sessions captured (every sealed Page of every session's last turn, a lower bound
    of the arena's peak), and the launcher's worst case (controller.pinned_host). A table; VmPin that never shows
    pinned memory (the CUDA host allocator does not count there: job 3343069 recorded 0) is the finding
    HOST_PIN_NOT_VISIBLE, a peak VmPin above the model plus 2 GiB the finding HOST_PIN_ABOVE_PREDICTED. Never a
    failure."""
    import controller
    table, hidden, above = [], [], []
    runs = {r['label']: r for r in options['runs']}
    for label, summary in by_label.items():
        run = runs.get(label)
        if run is None:
            continue
        layout = controller.layout_of(pins, run['layout'])
        P = layout['page_tokens']
        seqs = summary.get('running') or summary.get('sessions') or 1
        last = {}
        for turn in summary.get('turns') or ():
            stage = turn.get('stage_b') or {}
            if type(stage.get('computed_end')) is int:
                last[turn['session']] = max(last.get(turn['session'], 0), stage['computed_end'])
        pages = sum(end // P for end in last.values())
        page_bytes = P * FA_TOKEN_BYTES
        chunk = max(1, ARENA_CHUNK_BYTES // page_bytes) if layout['decoupled'] else 1
        arena = -(-pages // chunk) * chunk * page_bytes
        model = CONNECTOR_BYTES + seqs * CARRY_H2D_BYTES + arena
        worst = controller.pinned_host(pins, controller.run_options(options, dict(run, sessions=run['sessions'])),
                                       run['layout'], seqs)
        res = summary.get('resources') or {}
        gib = lambda v: None if v is None else round(v / GIB_BYTES, 2)
        row = dict(label=label, sessions=seqs, page_tokens=P, captured_pages=pages, arena_chunks_bytes=arena,
                   model_pinned_gib=gib(model), launcher_worst_case_gib=worst['gib'],
                   vmpin_max_gib=gib(res.get('host_vmpin_bytes_max')), vmlck_max_gib=gib(res.get('host_vmlck_bytes_max')),
                   vmrss_at_ready_gib=gib(res.get('host_vmrss_bytes_at_ready')), vmrss_max_gib=gib(res.get('host_vmrss_bytes_max')),
                   vmhwm_max_gib=gib(res.get('host_vmhwm_bytes_max')),
                   cgroup_current_max_gib=gib(res.get('cgroup_memory_current_bytes_max')),
                   cgroup_peak_gib=gib(res.get('cgroup_memory_peak_bytes_max')),
                   cgroup_limit_gib=gib(res.get('cgroup_memory_max_bytes')),
                   node_mem_available_min_gib=gib(res.get('node_mem_available_bytes_min')))
        if res.get('host_vmrss_bytes_at_ready') is not None and res.get('host_vmrss_bytes_max') is not None:
            row['vmrss_growth_after_ready_gib'] = gib(res['host_vmrss_bytes_max'] - res['host_vmrss_bytes_at_ready'])
        table.append(row)
        pin = res.get('host_vmpin_bytes_max')
        if not pin:
            hidden.append(label)
        elif pin > model + HOST_PIN_SLACK_BYTES:
            above.append(label)
    detail = dict(table=table, slack_gib=HOST_PIN_SLACK_BYTES // GIB_BYTES, vmpin_hidden=hidden, above=above)
    if above:
        raise Finding(HOST_PIN_FINDING, detail)
    if hidden:
        raise Finding(HOST_PIN_HIDDEN, detail)
    return detail
