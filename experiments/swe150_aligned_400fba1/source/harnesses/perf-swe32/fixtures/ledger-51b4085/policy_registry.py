"""The page-selection policies of the selective-prefix runtime, each declared once (policy-registry-1112).

A policy is one ``PagePolicy`` subclass (its name, kind and properties, and for a fill policy its ``select``
function) plus one registration (``register(TheClass())``). Everything else reads policy names and properties from
here instead of naming policies: the launcher (``serve.py --policy`` choices and refusals), the startup config
(``remote_ra_http.config``), the selection contract (``remote_ra_live_v2.selection_contract``: the pins), the query
authority (``remote_ra_live_v2.query_selection``: the plan), the engine and frontend gates, capacity, phase 1, the
Goal machinery switches and the HTTP evidence. The CLI and HTTP surface, the startup records and the evidence keys
and values are those of 4bf2c40 for its eight policies (docs/policy-registry.md).

Properties (class attributes; ``docs/policy-registry.md`` has the table of the eight built-in policies):

- ``kind``: ``full`` (every Page), ``random`` (a random draw: random50's floor(n/2), random_fill's fill),
  ``query`` (Selector V2 ranks the remaining Pages by the cosine to a query vector), ``window`` (Selector V2 fills
  the remaining capacity in the policy's own order) or ``rule`` (the policy's rule decides the selection; no budget).
- ``selector_v2``: the policy runs the Selector V2 contract (a selection contract in the startup config, two-phase
  query turns, a signed query receipt per turn). False: full and random50 (the native path).
- ``query_vector``: the hidden state that ranks the Pages: ``'observation'`` (phase 1's observation vector),
  ``'goal'`` (a Goal forward of the Goal query text) or None (no query vector, no projection, no score).
- ``goal_text``: the policy's query is its Goal's text: the Goal cache and forward, goals.jsonl's texts, the
  Terminus 2 Goal inputs of chat-text (``chat_text_goal``).
- ``goal_engine``: the policy runs the Goal rewriter (the Goal engine): ``--goal-rewriter-model``,
  ``--goal-rewriter-batch``, ``--goal-rewriter-engine vllm``, ``--goal-overlap``. Every other policy builds the
  deterministic Goal (evidence only) and records a given rewriter as unused (``rewriter_unused``).
- ``budget``: ``none`` (full), ``half`` (random50: floor(n/2)), ``fixed`` (the contract's active budget:
  ``active_ratio`` or ``active_token_budget``) or ``rule`` (no budget: the capacity is the rule's selection; the
  contract has neither and switch threshold 0; ``--active-token-budget`` / ``--active-ratio`` are refused).
- ``contract_key``: the policy's own key in the selection contract (its pin, ``pin()``), always its name, or None.
  ``pin_options``: the launcher's selection options the pin takes (random_fill: ``fill_seed``).
- ``evidence_key``: the key of the policy's own record in the plan's evidence, the signed receipt's
  ``selection`` and the runtime row's ``query.selection`` (``fill``, ``window``) or None.
- ``reply_roles``: the policy tells a reply's tool call and reasoning from the tool responses by the reply spans
  of the committed stream (``--thinking-history strip`` refuses it).
- ``rewriter_unused``: startup.json's ``goal_rewriter.reason`` when a rewriter model is given but not used.
- ``order_id``: a fill policy's pinned order (the selector config's ``fill``, the record's ``order``).

Plug-in policies: ``window`` and ``rule`` policies run through the runtime's fill path (Selector V2's boundary,
mandatory and recent Pages, then the policy's ``select`` over the remaining candidates, its record under
``window``). A ``query`` plug-in (ledger-policy-1112) is ranked by Selector V2's cosine to the observation vector,
as observation_query, with one hook on the fill (``query_hook``): before the cosine fill it may protect candidate
Pages (taken first, inside the budget), exclude candidates (never filled) and name units (Page sets the cosine fill
takes whole or not at all); ``hooked_fill`` is that fill, on both selection paths, and the hook's record goes under
``hook``. A release plug-in is one class in its own module (``source-overlay/page_policies/``), registered by one
``'module:Class'`` entry in ``PLUGINS`` below; a test registers its class itself (``registered``). Token-level
policies are out of scope (docs/policy-registry.md, section 6).

- ``protocols``: the startup protocols a policy serves (``mswea_xml_v1``, ``kv_session_v1``, ``chat_text_v1``,
  ``mswea_xml_gemma4_v1``), None for every one (the eight built-in policies). The launcher and the startup config
  refuse the others (``serves``).

This module imports nothing outside the standard library at import time (serve.py reads it before the release is
on sys.path); the built-in policies' algorithms stay where they were (``remote_ra_live_v2.selection_contract``,
``remote_ra_growing.contract``) and are reached lazily from their classes.
"""
from __future__ import annotations

from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib

KINDS = ('full', 'random', 'query', 'window', 'rule')
BUDGETS = ('none', 'half', 'fixed', 'rule')
QUERY_VECTORS = (None, 'observation', 'goal')
# The kinds of the runtime's fill path (no query vector, no Goal engine; a plug-in's select).
PLUGIN_KINDS = ('window', 'rule')
# The kind of a query plug-in: the cosine to the observation vector with the policy's query_hook (hooked_fill).
HOOK_KINDS = ('query',)
# The launcher's selection options a plug-in's pin may take (serve.py --ledger-cap-percent).
PLUGIN_PIN_OPTIONS = ('ledger_cap_percent',)
# The startup protocols (startup.json protocol) a policy may name in ``protocols``.
PROTOCOLS = ('mswea_xml_v1', 'kv_session_v1', 'chat_text_v1', 'mswea_xml_gemma4_v1')
# The eight policies of 4bf2c40, in its order (serve.py POLICIES); registered below, before any plug-in.
BUILTIN = ('full', 'random50', 'goal_query', 'observation_query', 'random_fill', 'recent_fill', 'recent_tools',
           'akv_rule')
# Release plug-ins, one 'module:Class' entry each (a PagePolicy subclass in its own module, e.g.
# 'page_policies.oldest_first:OldestFirst'): imported and registered once, after the built-in policies (load).
PLUGINS = ('page_policies.ledger_query:LedgerQuery',)

# One message of a session's sealed history after the initial prompt (FillContext.messages, a query hook's input):
# role 'assistant' (a reply: [start, end) from its generation header through its closing boundary; call = its first
# tool-call token, reasoning before it; text = its action text (thinking on: the content after </think>);
# reasoning = its reasoning text or None) or 'user' (an observation or a format-error message: call None, text =
# the message text, reasoning None). Token positions are absolute.
HistoryMessage = namedtuple('HistoryMessage', 'role start end call text reasoning')


def _require(ok, message):
    """The runtime's own refusal (remote_ra_live_v2.contract.LiveV2Error), as the selection contract raises it."""
    if not ok:
        from remote_ra_live_v2.contract import require
        require(ok, message)


def _sc():
    from remote_ra_live_v2 import selection_contract
    return selection_contract


@dataclass(frozen=True)
class FillContext:
    """One active turn of a fill policy (kind ``window`` or ``rule``) after Selector V2's safety partition.

    Built on ``remote_ra_growing.selective_policy.PageSelectionContext``: ``select(context)`` returns a
    ``PageSelection`` (the PageSelectionPolicy protocol) whose ``logical_pages`` are the fill order, a permutation
    of ``candidates``; the runtime takes the first ``capacity - mandatory - len(recent)`` of them (Selector V2's
    budget loop: every Page costs one). Pages are ints in [0, num_pages); token positions are absolute.

    num_pages: the candidate catalog n (the sealed Pages before the observation); page_tokens: P;
    mandatory: Pages [0, mandatory) (the initial prompt); recent: the recent safety Pages (ascending);
    candidates: the remaining Pages (ascending); capacity: the turn's Page capacity (a rule policy: its own
    selection); local_turn / turn: the session's local turn and request ordinal; session_index: the session's
    startup index; turn_ends: the stable end of every completed local turn; observation_spans: the [o, e) of every
    continuing turn's new messages; stable_end: this turn's; calls / reasoning: ``page_classes``' two Page sets
    (recent_tools: CALL and REASONING Pages; a rule policy: its kept Pages); pin: the contract's pin value;
    page_vectors: the candidate Pages' vectors (compact int8 or float rows; read only).

    A query plug-in's ``query_hook`` gets the same context (``candidates``: the Pages left after the mandatory
    and recent ones, ascending; ``capacity``: the budget's) plus ``messages``, the sealed history after the
    initial prompt as HistoryMessage rows in order (a rebuilt session's re-encoded messages first), and
    ``state``, a dict the policy may keep across the session's turns (dropped on a rebuild or a close)."""
    policy: str
    num_pages: int
    page_tokens: int
    mandatory: int
    recent: tuple
    candidates: tuple
    capacity: int
    local_turn: int
    turn: int
    session_index: int
    turn_ends: list
    observation_spans: list
    stable_end: int
    calls: frozenset = frozenset()
    reasoning: frozenset = frozenset()
    pin: object = None
    page_vectors: dict = field(default=None, compare=False)
    messages: tuple = ()
    state: dict = field(default=None, compare=False)


@dataclass(frozen=True)
class QueryHook:
    """A query plug-in's adjustment of one turn's cosine fill (``PagePolicy.query_hook``; ``hooked_fill``).

    protected: candidate Pages taken first, in this order (they count against the fill's room, at most all of it);
    excluded: candidate Pages the fill never takes (the selection may stay below the capacity); units: disjoint
    tuples of candidate Pages, neither protected nor excluded, that the cosine fill takes whole (when the
    cosine order reaches the unit's first Page and all of it fits) or not at all; record: a callable
    ``record(selected, plain, fill)`` returning the policy's evidence record (``selected``: the turn's selected
    Pages, ``plain``: the plain cosine selection of the same turn, both frozensets; ``fill``: hooked_fill's
    counts), or None."""
    protected: tuple = ()
    excluded: frozenset = frozenset()
    units: tuple = ()
    record: object = field(default=None, compare=False)


def hooked_fill(order, room, hook):
    """Selector V2's greedy budget loop (every Page costs one) for a query plug-in.

    ``order``: the candidate Pages by cosine, best first (Selector V2's order); ``room``: capacity - mandatory -
    recent. Returns (taken, rest, fill): ``taken`` = the hook's protected Pages, then the cosine order without the
    excluded and already taken Pages, a unit (all of its Pages not yet taken) when its first Page comes up and it
    fits in what is left, else none of it, until ``room`` Pages are taken; ``rest`` = the other candidates in cosine
    order; ``fill`` = counts for the record (protected, filled, units_whole, units_skipped, unfilled)."""
    order = list(order)
    candidates = set(order)
    protected = list(hook.protected)
    units = [tuple(unit) for unit in hook.units]
    unit_of = {}
    for k, unit in enumerate(units):
        for page in unit:
            _require(page not in unit_of, 'query hook units must be disjoint')
            unit_of[page] = k
    _require(len(set(protected)) == len(protected) and set(protected) <= candidates and len(protected) <= room
             and set(hook.excluded) <= candidates and set(unit_of) <= candidates
             and not (set(protected) | set(unit_of)) & set(hook.excluded) and not set(protected) & set(unit_of),
             'a query hook protects at most its room of distinct candidate Pages, and units, protected and '
             'excluded Pages are disjoint candidates')
    taken, seen, done = list(protected), set(protected), set()
    whole = skipped = 0
    for page in order:
        if len(taken) >= room:
            break
        if page in seen or page in hook.excluded:
            continue
        k = unit_of.get(page)
        if k is None:
            taken.append(page)
            seen.add(page)
            continue
        if k in done:
            continue
        done.add(k)
        need = [p for p in units[k] if p not in seen]
        if len(taken) + len(need) <= room:
            taken.extend(need)
            seen.update(need)
            whole += 1
        else:
            skipped += 1
    skipped += len(set(range(len(units))) - done)
    rest = [page for page in order if page not in seen]
    fill = dict(room=room, protected=len(protected), filled=len(taken) - len(protected), units_whole=whole,
                units_skipped=skipped, unfilled=max(0, room - len(taken)))
    return taken, rest, fill


class PagePolicy:
    """A page-selection policy of the live runtime. The defaults are a plug-in fill policy's (kind ``window``):
    Selector V2 with a fixed budget, no query vector, no Goal engine, its pin ``{order: order_id}`` and its record
    under ``window``. A plug-in sets ``name`` and ``order_id`` and implements ``select``."""

    name = None
    kind = 'window'
    selector_v2 = True
    query_vector = None
    goal_text = False
    goal_engine = False
    budget = 'fixed'
    evidence_key = 'window'
    reply_roles = False
    pin_options = ()
    order_id = None
    builtin = False
    protocols = None

    @property
    def contract_key(self):
        return self.name if self.kind in PLUGIN_KINDS or not self.builtin else None

    @property
    def rewriter_unused(self):
        return None if self.goal_engine else '%s fills the remaining Pages in its own order and ranks nothing' % self.name

    # ---- the selection contract pin (contract_key) ---------------------------------------------------------
    def pin(self, **options):
        """The contract value of this policy's key (``default_contract(**{name: pin()})``)."""
        _require(not options, '%s takes no pin options' % self.name)
        return self.validate_pin(dict(order=self.order_id))

    def validate_pin(self, value):
        _require(type(value) is dict and set(value) == {'order'} and value['order'] == self.order_id,
                 '%s pin {order: %r} required' % (self.name, self.order_id))
        return value

    def pin_required(self):
        """for_policy's refusal of this policy's contract without its pin."""
        return '%s requires the %s pin of its selection contract' % (self.name, self.name)

    def pin_foreign(self, policy):
        """for_policy's refusal of another policy's contract that carries this policy's pin."""
        return '%s does not fill by %s; its selection contract must not pin %s' % (policy, self.name, self.name)

    def budget_refusal(self):
        """serve.py's refusal of --active-token-budget / --active-ratio (budget ``rule``)."""
        return '--active-token-budget / --active-ratio do not apply to %s: its capacity is its rule' % self.name

    def info(self, pin):
        """The pin's fields of the kv-session info options (GET /v1/kv-session/info)."""
        return dict(fill_order=pin['order'])

    # ---- the fill (kinds window and rule) ------------------------------------------------------------------
    def page_classes(self, view):
        """(calls, reasoning): two Page sets in [view.mandatory, view.n) for ``select`` and ``record``; a rule
        policy's first set is its selection (the kept Pages). ``view``: n, mandatory, page_tokens, turn_ends,
        observation_spans, stable_end, replies (the reply spans), pin, call_starts(replies)."""
        return frozenset(), frozenset()

    def select(self, context):
        """PageSelectionPolicy.select: the fill order of ``context.candidates`` (FillContext)."""
        raise NotImplementedError('%s: a fill policy implements select(context)' % self.name)

    def record(self, context, selected, order, filled):
        """The policy's evidence record (``evidence_key``): selection_contract.window_record's common fields
        with this policy's ``order_id``."""
        return _sc().window_record(self.name, context.num_pages, context.mandatory, selected, order, filled,
                                   context.turn_ends, context.observation_spans, context.stable_end,
                                   context.page_tokens, context.calls, context.reasoning, order_id=self.order_id)

    # ---- the query hook (kind query plug-ins) ---------------------------------------------------------------
    hooked = False      # True: Selector V2's cosine fill runs through query_hook / hooked_fill

    def query_hook(self, context):
        """A query plug-in's QueryHook for one active turn (FillContext with ``messages`` and ``state``)."""
        raise NotImplementedError('%s: a query plug-in implements query_hook(context)' % self.name)

    def __repr__(self):
        return '<%s %s>' % (type(self).__name__, self.name)


def page_selection(name, pages):
    """remote_ra_growing.selective_policy.PageSelection(name, pages) (the PageSelectionPolicy protocol's result)."""
    from remote_ra_growing.selective_policy import PageSelection
    return PageSelection(name, tuple(pages))


# ---- the eight built-in policies (4bf2c40) ----------------------------------------------------------------------

class Full(PagePolicy):
    """Every Page (the reference)."""
    name, kind, selector_v2, budget, evidence_key, builtin = 'full', 'full', False, 'none', None, True
    rewriter_unused = None


class Random50(PagePolicy):
    """The reviewed floor(n/2) SHA-256 draw (remote_ra_growing.selective_policy's RandomPageSelectionPolicy at
    fraction 1/2, seed 42), seeded by the native prefix identity; no Selector V2."""
    name, kind, selector_v2, budget, evidence_key, builtin = 'random50', 'random', False, 'half', None, True
    rewriter_unused = None
    fraction = .5

    def select_native(self, n, native_prefix):
        from remote_ra_growing.contract import select_random50
        return select_random50(n, native_prefix)


class _Query(PagePolicy):
    kind, evidence_key, builtin = 'query', None, True


class GoalQuery(_Query):
    """Selector V2 ranked by the cosine to the maintained Goal's query vector (a Goal forward)."""
    name, query_vector, goal_text, goal_engine = 'goal_query', 'goal', True, True


class ObservationQuery(_Query):
    """Selector V2 ranked by the cosine to phase 1's observation vector; the deterministic Goal (evidence only)."""
    name, query_vector = 'observation_query', 'observation'
    rewriter_unused = 'observation_query ranks Pages by the observation vector only'


class RandomFill(PagePolicy):
    """The control of the query policies: Selector V2's partition, the remaining capacity drawn at random
    (selection_contract.fill_order: SplitMix64 and Fisher-Yates keyed by the pinned seed, the session index and
    the request ordinal); its own plan path (query_selection.plan_random_fill) and record (``fill``)."""
    name, kind, evidence_key, builtin = 'random_fill', 'random', 'fill', True
    pin_options = ('fill_seed',)
    rewriter_unused = 'random_fill fills the remaining Pages at random and ranks nothing'

    @property
    def contract_key(self):
        return self.name

    @property
    def order_id(self):
        return _sc().RANDOM_FILL_RNG

    def pin(self, seed=None):
        sc = _sc()
        return sc.random_fill_pin(sc.DEFAULT_FILL_SEED if seed is None else seed)

    def validate_pin(self, value):
        return _sc().validate_random_fill(value)

    def pin_required(self):
        return 'random_fill requires the random fill pin of its selection contract'

    def pin_foreign(self, policy):
        return '%s does not fill at random; its selection contract must not pin a random fill' % policy

    def info(self, pin):
        return dict(fill_seed=pin['seed'])


class _RecentWindow(PagePolicy):
    """The recent-window policies (recent-fill): selection_contract.window_order / window_record."""
    builtin = True

    @property
    def order_id(self):
        return _sc().WINDOW_ORDERS[self.name]

    def pin(self, **options):
        _require(not options, '%s takes no pin options' % self.name)
        return _sc().window_pin(self.name)

    def validate_pin(self, value):
        return _sc().validate_window(self.name, value)

    def pin_foreign(self, policy):
        return '%s does not fill a recent window; its selection contract must not pin %s' % (policy, self.name)

    def select(self, context):
        return page_selection(self.name, _sc().window_order(self.name, context.candidates, context.calls,
                                                            context.reasoning))

    def record(self, context, selected, order, filled):
        return _sc().window_record(self.name, context.num_pages, context.mandatory, selected, order, filled,
                                   context.turn_ends, context.observation_spans, context.stable_end,
                                   context.page_tokens, context.calls, context.reasoning)


class RecentFill(_RecentWindow):
    """The candidates newest first: the mandatory Pages plus one contiguous window ending at the newest Page."""
    name = 'recent_fill'
    rewriter_unused = 'recent_fill fills the remaining Pages newest first and ranks nothing'


class RecentTools(_RecentWindow):
    """The CALL Pages newest first, then the TOOL Pages, then the REASONING Pages (cover, CALL before TOOL)."""
    name, reply_roles = 'recent_tools', True
    rewriter_unused = ('recent_tools fills the remaining Pages with tool calls, then tool responses, then reasoning, '
                       'each newest first, and ranks nothing')

    def page_classes(self, view):
        sc = _sc()
        calls, reasoning = sc.window_classes(view.replies, view.call_starts(view.replies), view.observation_spans,
                                             view.page_tokens, view.mandatory, view.n)
        return frozenset(calls), frozenset(reasoning)


class AkvRule(_RecentWindow):
    """AKV's tool-response rolling drop (arXiv 2610.28577): every reply and the latest 12 tool responses; no
    budget (selection_contract.akv_kept: the kept Pages are the selection)."""
    name, kind, budget, reply_roles = 'akv_rule', 'rule', 'rule', True
    rewriter_unused = 'akv_rule keeps every reply and the latest 12 tool responses and ranks nothing'

    @property
    def order_id(self):
        return _sc().AKV_RULE_ID

    def pin(self, **options):
        _require(not options, '%s takes no pin options' % self.name)
        return _sc().akv_pin()

    def validate_pin(self, value):
        return _sc().validate_akv(value)

    def pin_required(self):
        return 'akv_rule requires the akv_rule pin of its selection contract'

    def pin_foreign(self, policy):
        return '%s does not apply the AKV rule; its selection contract must not pin akv_rule' % policy

    def budget_refusal(self):
        return '--active-token-budget / --active-ratio do not apply to akv_rule: its capacity is the AKV rule'

    def info(self, pin):
        return dict(budget_mode='rule', akv_rule=dict(pin))

    def page_classes(self, view):
        kept = _sc().akv_kept(view.turn_ends, view.observation_spans, view.stable_end, view.page_tokens,
                              view.mandatory, view.n, view.pin['latest_tool_responses'])[0]
        return frozenset(kept), frozenset()


# ---- the registry ------------------------------------------------------------------------------------------------

_POLICIES = {}


def _check(policy):
    """Registration checks: one name, a known kind and budget, consistent properties; a plug-in is a fill policy."""
    name = policy.name
    problems = []
    if type(name) is not str or not name or not all(c.islower() or c.isdigit() or c == '_' for c in name):
        problems.append('name must be a nonempty lower_snake_case str')
    elif name in _POLICIES:
        problems.append('name already registered')
    if policy.kind not in KINDS:
        problems.append('kind must be one of %s' % (KINDS,))
    if policy.budget not in BUDGETS:
        problems.append('budget must be one of %s' % (BUDGETS,))
    if policy.query_vector not in QUERY_VECTORS:
        problems.append('query_vector must be one of %s' % (QUERY_VECTORS,))
    if (policy.kind == 'query') != (policy.query_vector is not None):
        problems.append('a query vector belongs to kind query only')
    if (policy.kind == 'rule') != (policy.budget == 'rule'):
        problems.append('kind rule and budget rule go together')
    if (policy.kind == 'full') != (policy.budget == 'none'):
        problems.append('kind full and budget none go together')
    if policy.selector_v2 != (policy.budget in ('fixed', 'rule')):
        problems.append('Selector V2 policies have a fixed or a rule budget')
    if (policy.goal_engine or policy.goal_text) and policy.query_vector != 'goal':
        problems.append('the Goal engine and Goal text belong to a Goal query vector')
    if policy.contract_key not in (None, name):
        problems.append('contract_key is the policy name or None')
    if policy.contract_key is not None and not policy.selector_v2:
        problems.append('a contract pin needs Selector V2')
    if policy.evidence_key not in (None, 'fill', 'window', 'hook'):
        problems.append('evidence_key is fill, window, hook or None')
    if policy.protocols is not None and (type(policy.protocols) is not tuple or not policy.protocols
                                         or not set(policy.protocols) <= set(PROTOCOLS)):
        problems.append('protocols is None or a nonempty tuple of %s' % (PROTOCOLS,))
    if policy.hooked != (not policy.builtin and policy.kind in HOOK_KINDS):
        problems.append('a query hook belongs to a query plug-in')
    if not policy.builtin and policy.kind in HOOK_KINDS:
        # A query plug-in (the scoring hook): the observation vector's cosine with its query_hook.
        if policy.query_vector != 'observation' or policy.goal_engine or policy.goal_text:
            problems.append('a query plug-in ranks by the observation vector (no Goal engine or text)')
        if policy.contract_key != name or policy.evidence_key != 'hook':
            problems.append('a query plug-in pins its own key and records under hook')
        if not set(policy.pin_options) <= set(PLUGIN_PIN_OPTIONS):
            problems.append('a query plug-in takes pin options of %s only' % (PLUGIN_PIN_OPTIONS,))
        if type(policy.order_id) is not str or not policy.order_id:
            problems.append('a plug-in policy names its pinned order (order_id)')
        if type(policy).query_hook is PagePolicy.query_hook:
            problems.append('a query plug-in implements query_hook(context)')
        if type(policy.rewriter_unused) is not str:
            problems.append('rewriter_unused must be a str')
    elif not policy.builtin:
        if policy.kind not in PLUGIN_KINDS:
            problems.append('a plug-in policy is of kind %s (the fill path) or query (the query hook)'
                            % ' or '.join(PLUGIN_KINDS))
        if policy.contract_key != name or policy.evidence_key != 'window' or policy.pin_options:
            problems.append('a plug-in policy pins its own key, records under window and takes no pin options')
        if type(policy.order_id) is not str or not policy.order_id:
            problems.append('a plug-in policy names its pinned order (order_id)')
        if type(policy).select is PagePolicy.select:
            problems.append('a plug-in policy implements select(context)')
        if type(policy.rewriter_unused) is not str:
            problems.append('rewriter_unused must be a str')
    if problems:
        raise ValueError('page-selection policy %r refused: %s' % (name, '; '.join(problems)))


def register(policy):
    """Register one PagePolicy instance (its name must be new); returns it."""
    if not isinstance(policy, PagePolicy):
        raise TypeError('register() takes a PagePolicy instance')
    _check(policy)
    _POLICIES[policy.name] = policy
    return policy


def unregister(name):
    """Remove a plug-in policy (tests); the built-in policies stay."""
    if name in BUILTIN:
        raise ValueError('built-in policy %r cannot be removed' % name)
    del _POLICIES[name]


def load(entries):
    """Register the 'module:Class' entries (PLUGINS, at import), each class instantiated once; returns them."""
    loaded = []
    for entry in entries:
        module, _, name = entry.partition(':') if type(entry) is str else ('', '', '')
        if not module or not name:
            raise ValueError("plug-in entry 'module:Class' required: %r" % (entry,))
        loaded.append(register(getattr(importlib.import_module(module), name)()))
    return tuple(loaded)


@contextmanager
def registered(policy):
    """A plug-in policy registered for the duration of a block (tests)."""
    register(policy)
    try:
        yield policy
    finally:
        unregister(policy.name)


def get(name):
    """The registered policy of this name, or None (any other value, unhashable ones included)."""
    return _POLICIES.get(name) if type(name) is str else None


def known(name):
    return get(name) is not None


def names():
    """Every registered policy name, built-in ones first (4bf2c40's order), then plug-ins in registration order."""
    return tuple(_POLICIES)


def policies():
    return tuple(_POLICIES.values())


def where(**attributes):
    """The names of the policies whose attributes equal ``attributes``, in registration order."""
    return tuple(p.name for p in _POLICIES.values()
                 if all(getattr(p, key) == value for key, value in attributes.items()))


def builtin_where(**attributes):
    """``where`` over the eight built-in policies only (the constants modules keep for compatibility)."""
    return tuple(name for name in where(**attributes) if name in BUILTIN)


def pinned():
    """The policies that pin their own key in the selection contract, in registration order."""
    return tuple(p for p in _POLICIES.values() if p.contract_key is not None)


def selector_v2(name):
    policy = get(name)
    return policy is not None and policy.selector_v2


def query_vector(name):
    policy = get(name)
    return None if policy is None else policy.query_vector


def goal_text(name):
    policy = get(name)
    return policy is not None and policy.goal_text


def goal_engine(name):
    policy = get(name)
    return policy is not None and policy.goal_engine


def reply_roles(name):
    policy = get(name)
    return policy is not None and policy.reply_roles


def serves(name, protocol):
    """The policy serves this startup protocol (``protocols`` None: every one)."""
    policy = get(name)
    return policy is not None and (policy.protocols is None or protocol in policy.protocols)


def protocol_refusal(name):
    """The launcher's and the startup config's refusal of a protocol the policy does not serve."""
    return '%s serves the protocol %s only' % (name, joined(get(name).protocols))


PROPERTIES = ('kind', 'selector_v2', 'query_vector', 'goal_text', 'goal_engine', 'budget', 'contract_key',
              'pin_options', 'evidence_key', 'reply_roles', 'rewriter_unused', 'order_id', 'protocols')


def describe():
    """Every registered policy as a JSON-able row, in registration order: its name, builtin, the properties
    above and its default pin (``pin()``; None without a contract key). For readers outside the runtime (the
    harnesses' plans, docs/policy-registry.md 7), e.g. ``python -I -B -c "import sys; sys.path.insert(0,
    '<release>/source'); import json, policy_registry; print(json.dumps(policy_registry.describe()))"``; the
    built-in pins and orders come from remote_ra_live_v2.selection_contract (stdlib only)."""
    rows = []
    for policy in _POLICIES.values():
        row = dict(name=policy.name, builtin=policy.builtin)
        row.update((key, list(value) if type(value) is tuple else value)
                   for key, value in ((key, getattr(policy, key)) for key in PROPERTIES))
        row['pin'] = None if policy.contract_key is None else policy.pin()
        rows.append(row)
    return rows


def joined(items):
    """'a', 'a and b', 'a, b and c' (refusal texts naming several policies)."""
    items = list(items)
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


for _policy in (Full(), Random50(), GoalQuery(), ObservationQuery(), RandomFill(), RecentFill(), RecentTools(),
                AkvRule()):
    register(_policy)
assert names() == BUILTIN
# The release plug-ins' names, in PLUGINS order (registered after the built-in policies).
PLUGIN_NAMES = tuple(policy.name for policy in load(PLUGINS))
