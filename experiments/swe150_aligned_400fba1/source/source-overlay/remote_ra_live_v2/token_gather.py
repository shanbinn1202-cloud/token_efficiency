"""Token gather: single history tokens on top of a query policy's Page selection (token-gather-1114).

CPU-tested only. Option B of the per-token design note: Selector V2 selects whole P-token Pages; a token picker adds
single tokens of the other candidate Pages (the teammate's condition E: protocol markers; C: the newest separator
tokens), and their K/V rows are gathered at restore into fresh FullAttention blocks.

Why the gathered attention is exact (Qwen3.6-27B, ``mswea_xml_v1``):

- K is stored after RoPE (``qwen3_next.py`` ``_project_qkv_gate`` returns the post-norm, post-RoPE q, k, v and
  ``forward`` writes k to the paged cache), so a gathered key keeps the RoPE phase of its logical position; q is
  rotated at its own logical position (``positions.py``: compact slots never replace RoPE positions), so q.k
  depends on the logical distance only (partial rotary 0.25 and the interleaved MRoPE sections change nothing).
- FlashAttention reads a request's keys through its block table up to the physical sequence length; the causal
  mask only separates the prefix (every block before the run) from the run. The order of prefix keys is
  irrelevant to softmax attention, so a gathered block anywhere in the prefix is exact.
- The 48 Gated-DeltaNet layers carry their recurrent state of the whole history (the carry); no selection, Page
  or gathered, touches them.

What the runtime cannot do: mask single slots. A prefix block is attended in full: an empty, zero or duplicated
slot changes the softmax (a zero key adds exp(0) to the denominator), so the gathered count is padded to whole
P-token blocks with real history tokens (``pad``: the newest candidate tokens that are attended by nothing else),
recorded in the receipt with the picks. Every slot of a gathered block holds a recorded token.

Budget (``shared_blocks_v1``): the gathered blocks share the fill's room with the Pages, in blocks: the fill
takes a candidate Page in Selector V2's cosine order when the Page and the gather of the picks outside the taken
Pages still fit (``fill``); a pick inside a taken Page costs nothing more. Picks beyond room x P tokens are
dropped (lowest priority first).

The contract pin (selection contract key ``token_gather``, present only when the server runs it, so every other
contract keeps its bytes and digest): {picker, markers, quota, pad, budget}. Pickers (``PICKERS``):
``markers_v1``: every occurrence of each marker in the candidate Pages (a marker is the text of a token sequence,
encoded with the session's tokenizer; an occurrence keeps its first token: ``<|im_start|>user`` keeps the
``<|im_start|>`` of every user message), newest first, at most ``quota`` (0: no quota); ``separators_v1``: the
newest ``quota`` separator tokens of the candidate Pages (a token whose decoded text is non-empty and only spaces,
tabs, newlines and . , ? ! ; :). Per-token Goal scoring is out of scope: Selector V2 has one vector per Page.

Phase 1: the session pins the gathered blocks of its last turn until its next reservation commits; phase 1's rows
are [R Pages][gathered][tail][fresh] with SparsePlan(Q, Q - |R|P - G), so phase 1 attends exactly what the
previous phase 2 attended plus the Pages it sealed. A reservation gathers its own blocks again (fresh blocks; the
previous ones are released at its commit).

Scope: goal_query and observation_query on ``mswea_xml_v1`` (Qwen), the decoupled layout at P = 16; refused with
Gemma 4, every other protocol and policy, the legacy layouts and ``--release-idle-pages`` (a gathered block has
no DRAM copy of its own; readmission would have to gather again: not implemented). Standard library only.

KV layout (``check_views``, worker startup): the FlashAttention layout of vLLM 0.24.0, [blocks, 2, P, KV heads,
head size] (block axis 0, token axis 2, both read from the loaded backend's get_kv_cache_shape), in either stride
order (NHD or HND: the runtime's OffloadingConnector makes vLLM use HND); ``TensorIO.gather_tokens`` indexes the
views by those axes and their own strides.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import types

from .contract import require

KEY = 'token_gather'                 # selection contract key and worker startup record
WORKER_KEY = 'token_gather'
PAD_RULE = 'newest_unattended_v1'
BUDGET_RULE = 'shared_blocks_v1'
PIN_KEYS = frozenset(('picker', 'markers', 'quota', 'pad', 'budget'))
PROTOCOLS = ('mswea_xml_v1',)
PAGE_TOKENS = 16
DEFAULT_MARKERS = ('</think>', '<|im_end|>', '<|im_start|>user')
SEPARATOR_CHARS = frozenset(' \t\n.,?!;:')
DEFAULT_SEPARATOR_QUOTA = 1024
QUOTA_LIMIT = 65536
MARKER_LIMIT = 16
MARKER_CHARS = 64


# ---- pickers -------------------------------------------------------------------------------------------------

class TokenPicker:
    """One token picker: its pinned id (``name``), its pin and its picks of one turn.

    ``picks(view)`` returns candidate token positions in priority order (before the quota). ``view``: stream (the
    session's committed token ids), end (L = n * P: the candidates are [0, L)), page_tokens, eligible (the
    candidate Pages outside the mandatory and recent ones), state (a dict kept for the session until its rebuild
    or close), encode (text -> token ids), separator (token id -> bool), pin."""
    name = None
    default_quota = 0

    def pin(self, markers=None, quota=None):
        require(markers is None, '%s takes no markers' % self.name)
        return self.validate(dict(picker=self.name, markers=None, quota=self.default_quota if quota is None
                                  else quota, pad=PAD_RULE, budget=BUDGET_RULE))

    def validate(self, value):
        require(type(value) is dict and set(value) == PIN_KEYS and value['picker'] == self.name
                and value['pad'] == PAD_RULE and value['budget'] == BUDGET_RULE
                and type(value['quota']) is int and 0 <= value['quota'] <= QUOTA_LIMIT,
                '%s pin {picker, markers, quota in [0, %d], pad: %r, budget: %r} required'
                % (self.name, QUOTA_LIMIT, PAD_RULE, BUDGET_RULE))
        return value

    def picks(self, view):
        raise NotImplementedError


class StreamIndex:
    """Ascending positions of a session's token stream that match ``match(stream, t)``; each token is examined
    once (the stream only grows within a session). ``width``: the longest match; the last width - 1 positions stay
    tentative until the stream is long enough to decide them."""

    def __init__(self, width, match):
        self.width, self.match = width, match
        self.positions, self.decided, self.length = [], 0, 0

    def update(self, stream):
        n = len(stream)
        require(n >= self.length, 'a session token stream only grows')
        while self.positions and self.positions[-1] >= self.decided:
            self.positions.pop()
        match = self.match
        self.positions.extend(t for t in range(self.decided, n) if match(stream, t))
        self.decided = max(self.decided, n - self.width + 1)
        self.length = n
        return self.positions


def newest(positions, end, page_tokens, eligible, quota):
    """The positions below ``end`` whose Page is eligible, newest first, at most ``quota`` (0: all)."""
    out = []
    i = bisect_left(positions, end) - 1
    while i >= 0 and (not quota or len(out) < quota):
        t = positions[i]
        if t // page_tokens in eligible:
            out.append(t)
        i -= 1
    return out


class MarkerPicker(TokenPicker):
    """Every occurrence of the pinned markers in the eligible Pages, newest first (teammate's condition E)."""
    name = 'markers_v1'

    def pin(self, markers=None, quota=None):
        return self.validate(dict(picker=self.name, markers=list(DEFAULT_MARKERS if markers is None else markers),
                                  quota=0 if quota is None else quota, pad=PAD_RULE, budget=BUDGET_RULE))

    def validate(self, value):
        markers = value.get('markers') if type(value) is dict else None
        require(type(markers) is list and 0 < len(markers) <= MARKER_LIMIT and len(set(markers)) == len(markers)
                and all(type(m) is str and 0 < len(m) <= MARKER_CHARS for m in markers),
                '%s markers: 1-%d distinct texts of at most %d characters' % (self.name, MARKER_LIMIT, MARKER_CHARS))
        return super().validate(dict(value, markers=None)) and value

    def patterns(self, view):
        state = view.state
        if 'patterns' not in state:
            patterns = []
            for text in view.pin['markers']:
                ids = tuple(view.encode(text))
                require(ids and all(type(i) is int and i >= 0 for i in ids),
                        'token gather marker %r encodes to no tokens' % text)
                patterns.append(ids)
            state['patterns'] = tuple(patterns)
        return state['patterns']

    def picks(self, view):
        state = view.state
        index = state.get('index')
        if index is None:
            patterns = self.patterns(view)
            first = {}
            for ids in patterns:
                first.setdefault(ids[0], []).append(ids)

            def match(stream, t):
                options = first.get(stream[t])
                if options is None:
                    return False
                return any(len(ids) == 1 or tuple(stream[t:t + len(ids)]) == ids for ids in options)
            index = state['index'] = StreamIndex(max(len(ids) for ids in patterns), match)
        positions = index.update(view.stream)
        return newest(positions, view.end, view.page_tokens, view.eligible, view.pin['quota'])


class SeparatorPicker(TokenPicker):
    """The newest ``quota`` separator tokens of the eligible Pages (the teammate's condition C, 1,024)."""
    name = 'separators_v1'
    default_quota = DEFAULT_SEPARATOR_QUOTA

    def validate(self, value):
        value = super().validate(value)
        require(value['markers'] is None and value['quota'] > 0, '%s pin: markers null, quota >= 1' % self.name)
        return value

    def picks(self, view):
        state = view.state
        index = state.get('index')
        if index is None:
            separator = view.separator
            index = state['index'] = StreamIndex(1, lambda stream, t: separator(stream[t]))
        positions = index.update(view.stream)
        return newest(positions, view.end, view.page_tokens, view.eligible, view.pin['quota'])


PICKERS = {}


def register_picker(picker):
    """Register a TokenPicker instance under its ``name`` (the release pickers below; a test registers its own)."""
    require(isinstance(picker, TokenPicker) and type(picker.name) is str and picker.name
            and picker.name not in PICKERS, 'a token picker needs a new name')
    PICKERS[picker.name] = picker
    return picker


def unregister_picker(name):
    require(name not in RELEASE_PICKERS, 'release token pickers stay registered')
    PICKERS.pop(name, None)


for _picker in (MarkerPicker(), SeparatorPicker()):
    register_picker(_picker)
RELEASE_PICKERS = tuple(PICKERS)


def picker_of(value):
    require(type(value) is dict and value.get('picker') in PICKERS, 'unknown token gather picker')
    return PICKERS[value['picker']]


def pin(picker, markers=None, quota=None):
    """The ``token_gather`` contract value of a picker (serve.py --token-gather)."""
    require(picker in PICKERS, 'unknown token gather picker %r' % (picker,))
    return PICKERS[picker].pin(markers=markers, quota=quota)


def validate_pin(value):
    return picker_of(value).validate(value)


def separator_text(text):
    """Whether a decoded token is a separator: non-empty and only spaces, tabs, newlines and . , ? ! ; :."""
    return type(text) is str and bool(text) and set(text) <= SEPARATOR_CHARS


# ---- the contract, the server and the startup config -----------------------------------------------------------

def policies():
    """The policies token gather serves: the built-in query policies (policy_registry kind query: goal_query and
    observation_query, the Goal or the observation cosine order). A query plug-in (a query hook) is refused."""
    import policy_registry
    return policy_registry.builtin_where(kind='query')


def check_contract(contract, policy):
    """A selection contract's token gather pin against its policy (selection_contract.for_policy)."""
    value = contract.get(KEY)
    if value is None:
        return None
    require(policy in policies(), 'token gather serves %s only' % ' and '.join(policies()))
    require(contract['page_tokens'] == PAGE_TOKENS and contract.get('vector_format') is not None,
            'token gather needs the decoupled KV layout at P = %d' % PAGE_TOKENS)
    return validate_pin(value)


def check_config(config):
    """Startup config refusals (remote_ra_http.config.load): the protocol, the model, --release-idle-pages and
    another attention backend (--triton-attn)."""
    contract = config.get('selection_contract') or {}
    if KEY not in contract:
        return None
    require(config.get('model_profile') is None, 'token gather serves Qwen3.6-27B only (Gemma 4 is refused)')
    require(config.get('protocol') in PROTOCOLS, 'token gather serves the protocol %s only' % ' and '.join(PROTOCOLS))
    require(config.get('release_idle_pages') is not True,
            'token gather excludes --release-idle-pages: a gathered block has no DRAM copy of its own')
    require(config.get('attention_backend') is None,
            'token gather needs the FlashAttention backend (its KV layout [blocks, 2, P, heads, head size]; '
            '--triton-attn is refused)')
    require(config.get('page_tokens') == PAGE_TOKENS and config.get('kv_layout') is not None,
            'token gather needs the decoupled KV layout at P = %d' % PAGE_TOKENS)
    return contract[KEY]


def loaded():
    """True when this process's startup config pins token gather (the HTTP runtime), else None."""
    import os
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    contract = load().get('selection_contract') or {}
    return True if KEY in contract else None


def worker_flag(startup):
    """Worker startup: the record the engine sent, checked once against the pinned config."""
    import os
    sent = startup.get(WORKER_KEY)
    require(sent is None or sent is True, 'token_gather worker record must be true or absent')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker token_gather differs from the pinned startup config')
    return sent


# The FullAttention KV layout token gather is validated on (the get_kv_cache_shape of vLLM 0.24.0's FlashAttention
# backend: what GPUModelRunner allocates and views_from_runner reads): logical [blocks, K/V (2), P, KV heads, head size],
# block axis 0, token axis 2. The stride order is the backend's get_kv_cache_stride_order: NHD (that order in
# memory) or HND ([blocks, 2, heads, P, head size] in memory: what vLLM uses here because the runtime's
# OffloadingConnector requires it). GPU job 3602601 (2026-11-14) refused at startup on the earlier, wrong check
# ([2, blocks, P, ...], block axis 1), which the CPU fixtures shared; the axes now come from the backend.
BLOCK_AXIS, KV_AXIS, TOKEN_AXIS = 0, 1, 2
STRIDE_ORDERS = {(0, 1, 2, 3, 4): 'NHD', (0, 1, 3, 2, 4): 'HND'}


def backend_layout(backend, page_tokens, num_kv_heads, head_size, cache_dtype='auto'):
    """A FullAttention backend's KV layout at P from its static ``get_kv_cache_shape`` (no vLLM config is read),
    found as views_from_runner finds the block axis: the shape at 17 blocks, the block axis (the one axis that moves
    from 17 to 19 blocks) and the token axis (the one axis that moves from P to 2P tokens per block)."""
    def shape(blocks, tokens):
        return tuple(int(n) for n in backend.get_kv_cache_shape(blocks, tokens, num_kv_heads, head_size,
                                                                cache_dtype_str=cache_dtype))
    base, axes = shape(17, page_tokens), []
    for other in (shape(19, page_tokens), shape(17, 2 * page_tokens)):
        moved = [i for i, (a, b) in enumerate(zip(base, other)) if a != b]
        require(len(other) == len(base) and len(moved) == 1,
                'token gather: the attention backend\'s block or token axis is not uniquely identifiable')
        axes.append(moved[0])
    return dict(shape=base, block_axis=axes[0], token_axis=axes[1], heads=int(num_kv_heads), head_size=int(head_size))


def runner_layouts(runner, page_tokens):
    """{layer name: backend_layout} of every FullAttention layer of the loaded runner (its attention groups, as
    views_from_runner walks them; the loaded backend of each group)."""
    out = {}
    for group in runner._kv_cache_spec_attn_group_iterator():
        spec = group.kv_cache_spec
        if type(spec).__name__ != 'FullAttentionSpec':
            continue
        layout = backend_layout(group.backend, page_tokens, spec.num_kv_heads, spec.head_size,
                                runner.cache_config.cache_dtype)
        for name in group.layer_names:
            require(out.get(name, layout) == layout, 'token gather: layer %s has two attention layouts' % name)
            out[name] = layout
    return out


def stride_order(tensor, block_axis):
    """The physical order of a FullAttention tensor's axes (outermost first) read from its strides, or None when it
    is not dense within a block (the block stride alone may be padded)."""
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    inner = sorted((d for d in range(len(shape)) if d != block_axis), key=lambda d: (-strides[d], d))
    step = 1
    for d in reversed(inner):
        if shape[d] > 1 and strides[d] != step:
            return None
        step *= shape[d]
    return (block_axis,) + tuple(inner) if strides[block_axis] >= step else None


def check_views(views, page_tokens, layouts):
    """Worker startup, once: every FullAttention view is the KV layout token gather is validated on; returns the
    (block axis, token axis) of each FA tensor, {(layer, tensor index): (block axis, token axis)}, for
    ``TensorIO.attach_token_gather`` (``gather_tokens`` indexes by those axes and the tensors' own strides).

    ``layouts``: {layer: backend_layout} of the loaded runner (runner_layouts). Required per FA view: one kernel block
    per Page (ratio 1, kernel tokens P) and one tensor; the backend's logical layout FlashAttention's
    [blocks, 2, P, KV heads, head size] (block axis 0, K/V axis 1, token axis 2); the view's block axis the backend's
    (views_from_runner) and its shape the backend's at its block count; its strides one of the two stride orders
    (``STRIDE_ORDERS``: NHD or HND), dense within a block (the block stride may be padded)."""
    attention = [v for v in views if v.kind == 'attention']
    require(attention and type(layouts) is dict, 'token gather needs the FullAttention views and their layouts')
    axes = {}
    for v in attention:
        layout = layouts.get(v.name)
        require(layout is not None, 'token gather: no attention backend layout for %s' % v.name)
        require(v.ratio == 1 and v.kernel_tokens == page_tokens and len(v.tensors) == 1,
                'token gather needs one %d-token kernel block per Page and one tensor per FullAttention view (%s)'
                % (page_tokens, v.name))
        t, a = v.tensors[0], v.axes[0]
        expected = list(layout['shape'])
        expected[layout['block_axis']] = t.shape[a]
        require(layout['shape'] == (17, 2, page_tokens, layout['heads'], layout['head_size'])
                and (layout['block_axis'], layout['token_axis']) == (BLOCK_AXIS, TOKEN_AXIS)
                and a == layout['block_axis'] and tuple(t.shape) == tuple(expected),
                'token gather needs FullAttention views [blocks, 2, %d, heads, head size] (block axis 0, token axis 2: '
                'vLLM 0.24.0 FlashAttention); %s is %s with block axis %d, its backend %s with block axis %d and token '
                'axis %d' % (page_tokens, v.name, tuple(t.shape), a, layout['shape'], layout['block_axis'],
                             layout['token_axis']))
        order = stride_order(t, a)
        require(order in STRIDE_ORDERS,
                'token gather needs the NHD or HND stride order, dense within a block; %s has strides %s'
                % (v.name, tuple(t.stride())))
        axes[(v.name, 0)] = (a, layout['token_axis'])
    return axes


# ---- one turn: picks, fill, pad, record ------------------------------------------------------------------------

def blocks(tokens, page_tokens):
    return -(-tokens // page_tokens)


def fill(order, room, picks, page_tokens):
    """(taken Pages in fill order, gathered picks ascending, picks kept) of the shared-blocks fill.

    ``order``: the candidate Pages outside the mandatory and recent ones, best first (Selector V2's cosine order);
    ``room``: capacity - mandatory - recent, in blocks; ``picks``: token positions in those Pages, priority order.
    The picks beyond room x P are dropped. Walking ``order``, a Page is taken when it and the gather of the kept
    picks outside the taken Pages still fit: len(taken) + 1 + ceil(|picks left| / P) <= room. Without picks this is
    Selector V2's fill (the first ``room`` Pages)."""
    P = page_tokens
    require(type(room) is int and room >= 0 and type(P) is int and P > 0, 'token gather fill geometry')
    kept = list(picks)[:room * P]
    require(len(set(kept)) == len(kept), 'token gather picks must be distinct')
    count = Counter(t // P for t in kept)
    left, taken = len(kept), []
    for page in order:
        if len(taken) >= room:
            break
        c = count.get(page, 0)
        if len(taken) + 1 + blocks(left - c, P) <= room:
            taken.append(page)
            left -= c
    chosen = set(taken)
    gathered = sorted(t for t in kept if t // P not in chosen)
    require(len(taken) + blocks(len(gathered), P) <= room, 'token gather exceeds its room')
    return taken, gathered, kept


def pad(count, end, page_tokens, attended, gathered):
    """``count`` pad tokens: the newest candidate tokens below ``end`` whose Page is not attended and that are not
    gathered already (``PAD_RULE``). There are always enough: the unattended tokens minus the gathered ones are
    empty only when the gathered tokens fill whole Pages."""
    P, out, chosen = page_tokens, [], set(gathered)
    page = end // P - 1
    while len(out) < count and page >= 0:
        if page not in attended:
            for t in range((page + 1) * P - 1, page * P - 1, -1):
                if t not in chosen:
                    out.append(t)
                    if len(out) == count:
                        break
        page -= 1
    require(len(out) == count, 'token gather has no unattended token to pad its last block')
    return out


def plan(order, room, picks, end, page_tokens, attended):
    """One turn's token gather: (taken Pages, receipt gather {tokens, pad}, record counts).

    ``attended``: the mandatory and recent Pages (the taken ones are added); ``end``: L."""
    P = page_tokens
    taken, gathered, kept = fill(order, room, picks, P)
    extra = (-len(gathered)) % P
    padding = pad(extra, end, P, set(attended) | set(taken), gathered) if extra else []
    tokens = sorted(gathered + padding)
    plain = set(list(order)[:room])
    chosen = set(taken)
    record = dict(picks=len(kept), truncated=len(picks) - len(kept), covered=len(kept) - len(gathered),
                  gathered=len(gathered), pad=len(padding), blocks=len(tokens) // P,
                  plain_retrieved=len(plain), retrieved=len(taken), displaced=len(plain - chosen),
                  added=len(chosen - plain))
    return taken, dict(tokens=tokens, pad=sorted(padding)), record


def turn(contract, stream, end, page_tokens, order, room, attended, state, encode, separator):
    """The picks and plan of one active turn (query_selection), or None when the contract has no token gather.

    Returns (taken Pages, receipt gather or None, the evidence record)."""
    value = contract.get(KEY)
    if value is None:
        return None
    picker = picker_of(value)
    require(stream is not None and len(stream) >= end, 'token gather needs the committed token stream')
    eligible = frozenset(order)
    view = types.SimpleNamespace(stream=stream, end=end, page_tokens=page_tokens, eligible=eligible,
                                 state=state, encode=encode, separator=separator, pin=value)
    picks = picker.picks(view)
    require(all(type(t) is int and 0 <= t < end and t // page_tokens in eligible for t in picks)
            and len(set(picks)) == len(picks), '%s must pick distinct tokens of its candidate Pages' % picker.name)
    taken, gather, record = plan(order, room, picks, end, page_tokens, attended)
    record = dict(picker=value['picker'], **record)
    return taken, (gather if gather['tokens'] else None), record


# ---- the reservation: the native plan's gather and the worker's sources ------------------------------------------

def plan_entry(gather, selected, candidate_end, page_tokens):
    """The native plan's ``gather`` from a receipt's (adapter.reserve): {tokens, blocks}, checked."""
    if gather is None:
        return None
    P = page_tokens
    require(type(gather) is dict and set(gather) == {'tokens', 'pad'}, 'token gather receipt {tokens, pad}')
    tokens, padding = gather['tokens'], gather['pad']
    pages = set(selected)
    require(type(tokens) in (list, tuple) and tokens and all(type(t) is int for t in tokens) and
            list(tokens) == sorted(set(tokens)) and 0 <= tokens[0] and tokens[-1] < candidate_end and
            len(tokens) % P == 0 and not any(t // P in pages for t in tokens) and
            type(padding) in (list, tuple) and set(padding) <= set(tokens) and list(padding) == sorted(set(padding)),
            'token gather tokens: ascending, distinct, whole blocks, below L and outside the selected Pages')
    return dict(tokens=list(tokens), blocks=len(tokens) // P)


def dropped(candidate_end, selected, page_tokens, gather):
    """FA tokens left out of a phase-2 row: L - |A_t| P - G (G: the gathered tokens, whole blocks)."""
    G = 0 if gather is None else len(gather['tokens'])
    return candidate_end - len(selected) * page_tokens - G


def sources(tokens, page_tokens, resident, tail, catalog):
    """Where every gathered token's K/V is read from: ('device', {FA group: (block,)}, offset) for a token of a
    Page on the GPU (a resident Page, or the tail Page phase 1 completed: ``tail`` (page, blocks) or None), else
    ('host', its DRAM Page payload, offset)."""
    P, out = page_tokens, []
    for t in tokens:
        page, offset = divmod(t, P)
        if page in resident:
            out.append(('device', resident[page], offset))
        elif tail is not None and page == tail[0]:
            out.append(('device', tail[1], offset))
        else:
            require(page in catalog, 'gathered token %d has no DRAM Page' % t)
            out.append(('host', catalog[page], offset))
    return out


def cost(tokens, layers=16, kv_heads=4, head_size=256, dtype_bytes=2):
    """Bytes the gather moves: K and V of every FullAttention layer per token (Qwen3.6-27B: 16 x 2 x 4 x 256 x 2
    = 64 KiB per token, a 16-token Page 1 MiB)."""
    return tokens * layers * 2 * kv_heads * head_size * dtype_bytes
