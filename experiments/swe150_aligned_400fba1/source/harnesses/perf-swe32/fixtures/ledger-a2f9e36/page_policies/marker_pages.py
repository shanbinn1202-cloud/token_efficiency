"""goal_query_markers and observation_query_markers: the Pages holding protocol markers first (marker-pages-1114).

Some page-selection methods keep every earlier occurrence of a few protocol marker tokens (the end of a reasoning
block, the end of a message, the start and end of a tool response) on top of a token-level ranking under a fixed token
budget. The runtime restores whole Pages of P = 16 tokens (FlashAttention's smallest kernel block in vLLM 0.24), so
these two policies keep the Pages that hold a marker instead (the page approximation; docs/policy-registry.md,
section 9): goal_query or observation_query whose cosine fill first takes, through the registry's query hook
(policy_registry.QueryHook), every candidate Page that holds a marker occurrence, newest first.

- The candidates are the Pages outside the mandatory Pages (the initial prompt) and the recent safety Pages; markers
  inside those are attended anyway and only counted.
- The protected Pages are at most ``marker_cap_percent`` of the fill's room (B = capacity - mandatory - recent
  Pages; no cap: all of B), the newest kept; the record counts the dropped ones. They are inside the budget: the
  cosine fill (the Goal vector's for goal_query_markers, the observation vector's for observation_query_markers)
  takes the rest of B as the base policy does, and may still take a dropped marker Page by its score.
- A marker is a token sequence. Its first token is the marker token, whose Page is protected; the tokens after it
  only qualify the occurrence: ``<|im_start|>user`` is an ``<|im_start|>`` that opens a user message, i.e. an
  observation of the Mini-SWE-Agent XML protocol (the analog of a tool-response start).
- ``--marker-set`` names a set of MARKER_SETS (default ``xml_markers``: ``</think>``, ``<|im_end|>`` and
  ``<|im_start|>user``) or lists marker texts, comma-separated, each a token in angle brackets optionally followed
  by a word (``</think>,<|im_end|>``). The server checks every marker against its tokenizer at startup
  (``startup``): 1 to 8 tokens, encoded the same inside text, the sequences distinct.

The pin ``{order, marker_set, markers, marker_cap_percent}`` is the policy's key of the selection contract (serve.py
--marker-set, --marker-cap-percent; the cap None: none). The per-turn record goes under ``query.selection.hook``: the
markers found, the marker Pages, the protected and dropped Pages, the markers kept, the budget share and how the fill
differs from the plain cosine fill of the same turn. The hook scans each token of the session's history once (the
scan is kept in the session's state, which a rebuild drops). Served on the Qwen XML protocol only (``mswea_xml_v1``):
kv-session, chat-text and the Gemma 4 XML grammar are refused. Standard library only at import time.
"""
from __future__ import annotations

import re

import policy_registry as R

ORDER_ID = 'marker_pages_newest_first_v1'
PIN_KEYS = {'order', 'marker_set', 'markers', 'marker_cap_percent'}
# Named marker sets (the Qwen3.6 tokenizer's tokens; each is checked against the server's tokenizer at startup).
MARKER_SETS = {
    # The XML protocol's markers: the end of a reasoning block, the end of every message, the start of every
    # observation (the analog of a tool response's start on a native tool-call protocol).
    'xml_markers': ('</think>', '<|im_end|>', '<|im_start|>user'),
    'ends': ('</think>', '<|im_end|>'),
    'message_ends': ('<|im_end|>',),
    'reasoning_ends': ('</think>',),
    # The four markers of a native tool-call protocol; on mswea_xml_v1 the two tool-response tags never occur.
    'tool_markers': ('</tool_response>', '</think>', '<|im_end|>', '<tool_response>'),
}
DEFAULT_MARKER_SET = 'xml_markers'
CUSTOM = 'custom'
MAX_MARKERS = 8
MAX_MARKER_TOKENS = 8
# A custom marker: a token in angle brackets, optionally followed by a word (a role).
MARKER_RE = re.compile(r'<[^<>\s,]{1,62}>[A-Za-z_]{0,16}')
# The startup check: a marker encodes the same after each of these and before a newline.
CONTEXTS = ('x', ' ', '\n', '.')


def valid_markers(markers):
    return (type(markers) is list and 1 <= len(markers) <= MAX_MARKERS and len(set(markers)) == len(markers)
            and all(type(m) is str and MARKER_RE.fullmatch(m) for m in markers))


def marker_texts(value):
    """(marker_set, markers) of a ``--marker-set`` value: a named set (MARKER_SETS) or comma-separated marker texts
    (``custom``)."""
    R._require(type(value) is str and value.strip() != '',
               '--marker-set needs a marker set name or a comma-separated list of marker tokens')
    value = value.strip()
    if value in MARKER_SETS:
        return value, list(MARKER_SETS[value])
    markers = [m.strip() for m in value.split(',')]
    R._require(valid_markers(markers),
               "--marker-set %r is neither a marker set (%s) nor a list of marker tokens (comma-separated, at most %d, "
               "distinct, each a token in angle brackets optionally followed by a word, e.g. '</think>,<|im_end|>')"
               % (value, ', '.join(MARKER_SETS), MAX_MARKERS))
    return CUSTOM, markers


def contains(sequence, part):
    k = len(part)
    return any(sequence[i:i + k] == part for i in range(len(sequence) - k + 1))


def resolve(markers, encode, check=False):
    """Each marker's token ids (tuples, in order) by the server tokenizer's ``encode``. ``check`` (startup): each
    marker encodes the same inside text (after each of CONTEXTS, before a newline) and the sequences are distinct."""
    R._require(callable(encode), "marker pages need the server's tokenizer (the XML transcript's encode)")
    out = []
    for text in markers:
        ids = tuple(encode(text))
        R._require(1 <= len(ids) <= MAX_MARKER_TOKENS and all(type(t) is int and t >= 0 for t in ids),
                   'marker %r must encode to 1-%d tokens' % (text, MAX_MARKER_TOKENS))
        if check:
            for lead in CONTEXTS:
                R._require(contains(tuple(encode(lead + text + '\n')), ids),
                           'marker %r must encode the same inside text (after %r)' % (text, lead))
        out.append(ids)
    R._require(not check or len(set(out)) == len(out), 'markers must encode to distinct token sequences')
    return tuple(out)


def table_of(ids):
    """{first token: ((marker index, the marker's other tokens), ...)}."""
    table = {}
    for k, sequence in enumerate(ids):
        table.setdefault(sequence[0], []).append((k, sequence[1:]))
    return {first: tuple(entries) for first, entries in table.items()}


def occurrences(tokens, start, stop, table):
    """(position, marker index) of every marker whose first token lies in [start, stop), in position order; a
    marker occurs when its whole sequence is in ``tokens`` (two markers may share a position)."""
    found = []
    stop = min(stop, len(tokens))
    for first, entries in table.items():
        i = start
        while i < stop:
            try:
                i = tokens.index(first, i, stop)
            except ValueError:
                break
            for k, rest in entries:
                if not rest or tuple(tokens[i + 1:i + 1 + len(rest)]) == rest:
                    found.append((i, k))
            i += 1
    found.sort()
    return found


def scan(context, ids):
    """(the marker occurrences of the sealed history [0, num_pages * page_tokens), the tokens scanned now).

    The session's ``state['marker_scan']`` keeps the occurrences whose whole sequence lies before ``done`` (the
    history only grows between rebuilds, which drop the state); a turn scans the tokens past ``done`` only. The
    cache is checked at two tokens (the last scanned one and the newest occurrence) and rebuilt if they differ."""
    tokens, limit = context.tokens, context.num_pages * context.page_tokens
    R._require(len(tokens) >= limit, 'the token ids must cover the candidate Pages')
    table = table_of(ids)
    longest = max(len(sequence) for sequence in ids)
    state = context.state if context.state is not None else {}
    cache = state.get('marker_scan')
    if (cache is None or cache['ids'] != ids or cache['done'] > len(tokens)
            or (cache['done'] and tokens[cache['done'] - 1] != cache['last'])
            or (cache['found'] and tokens[cache['found'][-1][0]] != ids[cache['found'][-1][1]][0])):
        cache = state['marker_scan'] = dict(ids=ids, done=0, last=None, found=[])
    start = cache['done']
    done = max(start, min(limit, len(tokens) - longest + 1))
    if done > start:
        cache['found'].extend(occurrences(tokens, start, done, table))
        cache['done'], cache['last'] = done, tokens[done - 1]
    found = [x for x in cache['found'] if x[0] < limit]
    if limit > done:
        found += occurrences(tokens, done, limit, table)
    return found, max(0, limit - start)


def plan(context, pin):
    """The QueryHook of one turn: the candidate Pages holding a marker, newest first, up to the cap."""
    P, n = context.page_tokens, context.num_pages
    markers, cap_percent = pin['markers'], pin['marker_cap_percent']
    state = context.state
    key = tuple(markers)
    ids = (state.get('marker_ids') or {}).get(key) if state is not None else None
    if ids is None:
        ids = resolve(markers, context.encode)
        if state is not None:
            state['marker_ids'] = {key: ids}
    found, scanned = scan(context, ids)
    candidates = set(context.candidates)
    room = max(0, context.capacity - context.mandatory - len(context.recent))
    cap = room if cap_percent is None else room * cap_percent // 100
    counts = [0] * len(markers)
    for _, k in found:
        counts[k] += 1
    positions = sorted({t for t, _ in found})
    per_page = {}                       # candidate Page -> marker tokens on it
    for t in positions:
        if t // P in candidates:
            per_page[t // P] = per_page.get(t // P, 0) + 1
    newest = sorted(per_page, reverse=True)
    protected, dropped = newest[:cap], newest[cap:]

    def record(selected, plain, fill):
        selected, plain = set(selected), set(plain)
        low = context.mandatory
        mine = {p for p in selected if low <= p < n}
        theirs = {p for p in plain if low <= p < n}
        return dict(pin, room_pages=room, cap_pages=cap, occurrences=list(counts), marker_tokens=len(positions),
                    fixed_tokens=len(positions) - sum(per_page.values()), marker_pages=len(newest),
                    protected_pages=len(protected), dropped_pages=len(dropped),
                    dropped_tokens=sum(per_page[p] for p in dropped),
                    dropped_filled=sum(1 for p in dropped if p in selected),
                    kept_tokens=sum(1 for t in positions if t // P in selected),
                    budget_share=round(len(protected) / context.capacity, 4) if context.capacity else 0.0,
                    room_share=round(len(protected) / room, 4) if room else 0.0,
                    new_pages=sum(1 for p in protected if p not in plain), displaced_pages=len(theirs - mine),
                    fill=dict(fill), scanned_tokens=scanned,
                    plain_jaccard=round(len(mine & theirs) / len(mine | theirs), 4) if mine | theirs else 1.0)
    return R.QueryHook(protected=tuple(protected), record=record)


class _MarkerPages(R.PagePolicy):
    """A query policy whose cosine fill first takes the candidate Pages holding a marker (module docstring)."""
    kind = 'query'
    evidence_key = 'hook'
    hooked = True
    order_id = ORDER_ID
    pin_options = ('marker_set', 'marker_cap_percent')
    protocols = ('mswea_xml_v1',)
    # reply_roles stays False (as ledger_query's): it only selects --thinking-history strip's refusal, and strip
    # serves chat-text, which these policies do not serve.

    def pin(self, marker_set=None, marker_cap_percent=None, **options):
        R._require(not options, '%s takes only marker_set and marker_cap_percent' % self.name)
        name, markers = marker_texts(DEFAULT_MARKER_SET if marker_set is None else marker_set)
        return self.validate_pin(dict(order=ORDER_ID, marker_set=name, markers=markers,
                                      marker_cap_percent=marker_cap_percent))

    def validate_pin(self, value):
        cap = value.get('marker_cap_percent') if type(value) is dict else None
        R._require(type(value) is dict and set(value) == PIN_KEYS and value['order'] == ORDER_ID
                   and type(value['marker_set']) is str and valid_markers(value['markers'])
                   and (value['marker_set'] == CUSTOM
                        or list(MARKER_SETS.get(value['marker_set'], ())) == value['markers'])
                   and (cap is None or (type(cap) is int and 1 <= cap <= 100)),
                   '%s pin {order: %r, marker_set: one of %s or %s, markers: its marker texts, marker_cap_percent: '
                   'null or 1-100} required' % (self.name, ORDER_ID, ', '.join(MARKER_SETS), CUSTOM))
        return value

    def pin_foreign(self, policy):
        return '%s does not protect marker Pages; its selection contract must not pin %s' % (policy, self.name)

    def info(self, pin):
        return dict(fill_order=pin['order'], marker_set=pin['marker_set'], marker_cap_percent=pin['marker_cap_percent'])

    def startup(self, pin, encode):
        """The markers against the server's tokenizer (resolve's startup check)."""
        resolve(self.validate_pin(pin)['markers'], encode, check=True)

    def query_hook(self, context):
        return plan(context, self.validate_pin(context.pin))


class GoalQueryMarkers(_MarkerPages):
    """goal_query (the Goal vector's cosine, the Goal engine and text) with the marker Pages first."""
    name = 'goal_query_markers'
    query_vector, goal_text, goal_engine = 'goal', True, True


class ObservationQueryMarkers(_MarkerPages):
    """observation_query (the observation vector's cosine) with the marker Pages first."""
    name = 'observation_query_markers'
    query_vector = 'observation'
    rewriter_unused = ('observation_query_markers ranks Pages by the observation vector after its marker Pages and '
                       'builds no Goal query')
