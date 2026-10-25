"""Server-owned query preparation, upstream Selector V2 plan and one-use authorization.

Only the model-forward driver is an unqualified loaded seam. This module never
accepts a client vector, ranking, query receipt or issue override. The selection
semantics are the dram-backed-kv Online Selector V2 contract in
``selection_contract``.

Page vectors come from the worker: normalised float rows that this module
compacts (legacy), or, under a ``compact_int8_v1`` contract (decoupled KV
layout), int8 vectors the worker compacted on the device; then nothing here
projects a Page and the catalog holds only the compact vectors.

``random_fill`` (the control of the query policies) runs the same turns, Goal
(deterministic, evidence only) and Selector V2 safety partition as
observation_query, but takes no query vector: the fill step walks a seeded
random permutation of the remaining candidates (``random_fill_selector``).

Long histories (compact_int8_v1, P = 16): ``plan`` and ``plan_random_fill``
take a vectorized path (``fast_selection``) that computes the same partition,
the same cosine floats, the same tie-break and the same random fill with numpy
instead of Page records, the upstream index and per-Page Python loops; the
receipt and evidence are identical. It is enabled at startup only if numpy is
present and its self-check reproduces the reference floats and permutations;
otherwise, for the legacy float representation, and in any state the fast path
does not model, the reference path (the upstream selector) runs.
"""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import importlib
from itertools import repeat
import json
import math
from pathlib import Path
import secrets
import sys
import time
import types

from .contract import require
from . import fast_selection
from . import goal_text
from . import selection_contract


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value): return sha256(canonical(value)).hexdigest()


def latest_action(conversation):
    """Extract action from renderer-normalized messages without losing tool calls.

    Call IDs bind responses but do not change action semantics. Parsed arguments
    are canonicalized with the function name; message content remains available.
    Legacy content-only actions retain their original representation.
    """
    require(len(conversation)>1 and conversation[-2].get('role')=='assistant',
            'latest observation needs a preceding assistant action')
    assistant,observation=conversation[-2:]
    content=assistant.get('content') or ''
    require(type(content) is str,'normalized assistant content must be text')
    calls=assistant.get('tool_calls')
    if not calls:return content,{}
    require(type(calls) is list and len(calls)==1 and type(calls[0]) is dict,
            'one tool call per observation required')
    call=calls[0];function=call.get('function')
    require(call.get('type')=='function' and type(call.get('id')) is str and bool(call['id'])
            and call['id']==observation.get('tool_call_id'),'tool action/result ID mismatch')
    require(type(function) is dict and type(function.get('name')) is str
            and bool(function['name']) and type(function.get('arguments')) is dict,
            'renderer-normalized function name and parsed arguments required')
    action=dict(content=content,tool_calls=[dict(name=function['name'],arguments=function['arguments'])])
    return canonical(action).decode(),{'tool_name':function['name']}


def upstream():
    # Match the reviewed goal_adapter's namespace isolation; these are exact
    # packaged upstream modules, not a reimplementation of Goal or selector.
    root = Path(__file__).resolve().parents[1] / 'dram_kv'
    for name, path in [('dram_kv', root), ('dram_kv.goals', root/'goals'),
                       ('dram_kv.selectors', root/'selectors')]:
        if name not in sys.modules:
            module = types.ModuleType(name); module.__path__ = [str(path)]
            sys.modules[name] = module
    return (importlib.import_module('dram_kv.page_vectors'),
            importlib.import_module('dram_kv.schema'),
            importlib.import_module('dram_kv.goals.maintained'),
            importlib.import_module('dram_kv.selectors.prefill_v2'))


def goal_rewriter(pin, batch=None):
    """Start the upstream persistent Qwen Goal rewriter; fail closed if it cannot start.

    Individual rewrite failures still fall back to deterministic Goal fields
    inside the upstream constructor and are recorded per turn. ``batch``
    (--goal-rewriter-batch N, goal_rewriter_batch): the batched child, the calls
    that queue sent together up to N at a time; None: the upstream child, one
    call at a time.
    """
    if pin is None:
        return None
    importlib.import_module('dram_kv.goals.maintained')
    rewriter = importlib.import_module('dram_kv.goals.rewriter')
    source = Path(__file__).resolve().parents[1]
    manifest = json.loads((source.parent/'runtime-manifest.json').read_bytes())
    if batch is None:
        process = rewriter.PersistentCommandGoalRewriter(rewriter_argv(pin, source, manifest['site']),
            timeout_seconds=pin['timeout_seconds'],
            startup_timeout_seconds=pin['startup_timeout_seconds'])
        process.__enter__()
    else:
        from .goal_rewriter_batch import start
        process = start(rewriter, rewriter_argv(pin, source, manifest['site'], batch=batch), pin, batch)
    import atexit
    atexit.register(process.close)
    return process


def rewriter_argv(pin, source, site, executable=None, batch=None):
    """Upstream rewriter command, isolated to the sealed source and site paths; with
    ``batch`` (--goal-rewriter-batch N) the batched child (goal_rewriter_child) with the
    same arguments and ``--batch N``."""
    model = Path(pin['model'])
    require((model/'config.json').is_file() and sha256((model/'config.json').read_bytes()).hexdigest()
            == pin['config_sha256'], 'Goal rewriter checkpoint differs from its startup pin')
    program = ("import sys,runpy;sys.path[:0]=[sys.argv[1],sys.argv[2]];script=sys.argv[3];"
               "sys.argv=[script]+sys.argv[4:];runpy.run_path(script,run_name='__main__')")
    script = (Path(source)/'dram_kv_scripts'/'qwen_goal_rewriter.py' if batch is None else
              Path(source)/'remote_ra_live_v2'/'goal_rewriter_child.py')
    argv = (executable or sys.executable, '-I', '-S', '-B', '-u', '-c', program, str(source), str(site),
            str(script), '--model', str(model),
            '--device', 'cuda', '--dtype', 'bfloat16',
            '--max-new-tokens', str(pin['max_new_tokens']), '--local-files-only', '--serve')
    return argv if batch is None else argv + ('--batch', str(batch))


def normalized(values):
    require(type(values) in (tuple, list) and 0 < len(values) <= 65536
            and all(type(v) in (int, float) and math.isfinite(v) for v in values),
            'finite bounded native hidden vector required')
    scale = max(abs(v) for v in values)
    require(scale > 0, 'zero native hidden vector')
    norm = math.sqrt(sum((v/scale)**2 for v in values))
    return tuple((v/scale)/norm for v in values)


_MEMO_TYPES = {}


def memo_types(vectors):
    """Upstream CompactVector / PageVectorProvenance whose derived values are computed once.

    ``unit_values``, ``content_hash`` and ``provenance_hash`` are pure functions
    of frozen fields. The upstream selector reads them for every Page of every
    plan (snapshot reference check, cosine, reasons); at 16-token Pages that is
    ~4K Pages per turn. The subclasses keep every value and every comparison of
    the upstream classes (instances only ever meet instances of the same
    subclass); they only remember what was computed.
    """
    key = (vectors.CompactVector, vectors.PageVectorProvenance)
    if key not in _MEMO_TYPES:
        vector, provenance = key

        class MemoCompactVector(vector):
            def unit_values(self):
                memo = self.__dict__.get('_memo_unit')
                if memo is None:
                    memo = vector.unit_values(self)
                    object.__setattr__(self, '_memo_unit', memo)
                return memo

            @property
            def content_hash(self):
                memo = self.__dict__.get('_memo_content')
                if memo is None:
                    memo = vector.content_hash.fget(self)
                    object.__setattr__(self, '_memo_content', memo)
                return memo

        class MemoProvenance(provenance):
            @property
            def provenance_hash(self):
                memo = self.__dict__.get('_memo_provenance')
                if memo is None:
                    memo = provenance.provenance_hash.fget(self)
                    object.__setattr__(self, '_memo_provenance', memo)
                return memo

        _MEMO_TYPES[key] = (MemoCompactVector, MemoProvenance)
    return _MEMO_TYPES[key]


def _compact_wire(value):
    """A ``compact_int8_v1`` wire pair ([[q...], scale]) rather than a float row."""
    return type(value) in (list, tuple) and len(value) == 2 and type(value[0]) in (list, tuple)


# random_fill's selector reads no vector: its "vector space" is only the
# 16-token Page record size that the upstream Page check compares against.
_FILL_SPACE = types.SimpleNamespace(page_size=selection_contract.UPSTREAM_PAGE_TOKENS)
_RANDOM_FILL_TYPES = {}


def random_fill_selector(selectors):
    """Upstream ``PrefillPageEmbeddingSelectorV2`` with a random fill step (random_fill).

    ``select`` is the upstream one with the same boundary, Page-size and safety
    partition (mandatory, recent, candidates, capacity) and the same greedy
    budget loop; only the order of that loop differs: the candidates (in Page
    order) are permuted by ``selection_contract.fill_order`` under the pinned
    seed, the session index and the request ordinal, instead of being sorted by
    cosine. No snapshot, query or Page vector is read, and an unusable catalog
    fails closed instead of falling back to the lexical selector. ``order`` keeps
    the whole permutation (the receipt's ranking after mandatory and recent).
    """
    base = selectors.PrefillPageEmbeddingSelectorV2
    if base not in _RANDOM_FILL_TYPES:
        class PrefillPageRandomFillV2(base):
            name = 'prefill_page_random_fill_v2'

            def __init__(self, *, frozen_goal_trace_hash, page_budget, recent_turns, recent_quota,
                         seed, session_index, turn):
                super().__init__(None, _FILL_SPACE, frozen_goal_trace_hash=frozen_goal_trace_hash,
                                 page_budget=page_budget, recent_turns=recent_turns,
                                 recent_quota=recent_quota)
                self.fill = (seed, session_index, turn)
                self.order = None

            def _base_config(self):
                seed, session_index, turn = self.fill
                return {'selector_version': 'v2', 'page_size': self.provenance.page_size,
                        'page_budget': self.page_budget, 'active_token_budget': self.active_token_budget,
                        'recent_turns': self.recent_turns, 'recent_quota': self.recent_quota,
                        'query_source': None, 'fill': 'uniform_without_replacement',
                        'fill_rng': selection_contract.RANDOM_FILL_RNG, 'fill_seed': seed,
                        'fill_session_index': session_index, 'fill_turn': turn,
                        'frozen_goal_trace_hash': self.frozen_goal_trace_hash}

            def select(self, context, goal, catalog, previous_active=(), ever_selected=()):
                started = time.perf_counter()
                history = self._validate_boundary(context, goal, catalog)
                unsupported = self._physical_page_error(history)
                require(unsupported is None, 'random fill needs 16-token Page records: %s' % unsupported)
                mandatory, recent, candidates, used, capacity = self._safety_partition(context, history)
                order = selection_contract.fill_order(candidates, *self.fill)
                self.order = tuple(page.page_id for page in order)
                retrieved = []
                for page in order:
                    cost = self._cost(page)
                    if used + cost <= capacity:
                        retrieved.append(page)
                        used += cost
                mandatory_set = {page.page_id for page in mandatory}
                recent_set = {page.page_id for page in recent}
                retrieved_set = {page.page_id for page in retrieved}
                selected = mandatory_set | recent_set | retrieved_set
                ordered = tuple(page.page_id for page in history if page.page_id in selected)
                rank = {page.page_id: k for k, page in enumerate(order)}
                reasons = {page_id: selectors.SelectionReason(
                    source=('mandatory' if page_id in mandatory_set else
                            'recent_safety' if page_id in recent_set else 'random_fill'),
                    score=0.0, detail='' if page_id not in retrieved_set else 'fill_rank=%d' % rank[page_id])
                    for page_id in ordered}
                delta = selectors.active_set_delta(ordered, previous_active, ever_selected)
                plan = selectors.SelectionPlan(
                    session_id=context.session_id, turn_id=context.turn_id, goal_hash=goal.version_hash,
                    selector=self.name,
                    selector_config=self._complete_config({'effective_mode': 'random_fill',
                                                           'stable_tie_break': 'none'}),
                    ordered_page_ids=ordered, reasons=reasons,
                    mandatory_page_ids=tuple(p.page_id for p in history if p.page_id in mandatory_set),
                    recent_page_ids=tuple(p.page_id for p in history if p.page_id in recent_set),
                    retrieved_page_ids=tuple(p.page_id for p in history if p.page_id in retrieved_set),
                    active_page_budget=self.page_budget, active_token_budget=self.active_token_budget,
                    confidence=None, latency_ms=(time.perf_counter() - started) * 1_000,
                    added_page_ids=delta.added, retained_page_ids=delta.retained,
                    removed_page_ids=delta.removed)
                plan.validate_against(context, catalog)
                return plan

        _RANDOM_FILL_TYPES[base] = PrefillPageRandomFillV2
    return _RANDOM_FILL_TYPES[base]




_PAGE_KEYS = []


def page_keys(n):
    """``[str(0), ..., str(n - 1)]`` (the evidence keys), from one growing list."""
    if len(_PAGE_KEYS) < n:
        _PAGE_KEYS.extend(map(str, range(len(_PAGE_KEYS), n)))
    return _PAGE_KEYS[:n]


def _issue(task):
    require(type(task) is dict and set(task) == {'instance_id', 'problem_statement'}
            and all(type(v) is str and v.strip() for v in task.values()),
            'trusted startup issue required for query policies')
    return dict(task)


class QuerySelectionAuthority:
    # The reference selection path unless __init__ enabled the vectorized one.
    fast = None

    def __init__(self, adapter, worker=None, *, task, policy=None, contract=None, transcript=None,
                 sessions=None, rewriter_batch=None):
        """``sessions`` (multi-session stage 1) binds each session id to its own
        (trusted task, engine transcript); session 0's pair is ``task`` and
        ``transcript``. Without it every session reads ``task`` and ``transcript``.
        ``rewriter_batch`` (--goal-rewriter-batch N, goal_query with a pinned
        rewriter): the Goal rewriter's calls run batched (goal_rewriter_batch)."""
        # The server's policy decides the Goal rewriter. private_endpoint passes
        # the engine owner's policy (the startup config's); without it, the one
        # policy of the sessions the server opened.
        if policy is None:
            policies = {session.policy for session in adapter.sessions.values()}
            require(len(policies) == 1, 'server policy required')
            policy, = policies
        require(policy in ('full', 'random50', 'goal_query', 'observation_query', 'random_fill')
                and all(session.policy == policy for session in adapter.sessions.values()),
                'selection authority policy differs from the server policy')
        self.adapter, self.worker, self.task, self.policy = adapter, worker, _issue(task), policy
        # XML protocol (upstream Mini-SWE-Agent text actions) when a transcript exists.
        self.transcript = transcript
        self.trusted = None
        if sessions is not None:
            require(type(sessions) is dict and sessions and
                    next(iter(sessions.values())) == (self.task, transcript) and
                    all(type(pair) is tuple and len(pair) == 2 and
                        (pair[1] is None) == (transcript is None) for pair in sessions.values()),
                    'one trusted task and transcript per session required')
            self.trusted = {sid: (_issue(t), x) for sid, (t, x) in sessions.items()}
        # Startup only: each session's index (session order), random_fill's stream key.
        self.session_index = {sid: k for k, sid in enumerate(
            self.trusted if self.trusted is not None else adapter.sessions)}
        if contract is None and policy == selection_contract.RANDOM_FILL:
            contract = selection_contract.default_contract(
                adapter.page_tokens, random_fill=selection_contract.random_fill_pin())
        # Fails closed on a contract that pins the rewriter for observation_query
        # (or random_fill), and on a random fill pin for any other policy.
        self.contract = selection_contract.for_policy(dict(contract) if contract is not None
            else selection_contract.default_contract(adapter.page_tokens), policy)
        require(self.contract['page_tokens'] == adapter.page_tokens,
                'selection contract Page size differs from native Pages')
        self.contract_sha256 = selection_contract.digest(self.contract)
        self.vector_catalog, self.authorized, self.used = {}, {}, set()
        # Per session: stable boundary of every completed preparation (index =
        # vLLM turn), the resident set after the prior turn and every Page ever
        # selected (the last two only feed upstream churn diagnostics).
        self.turn_ends, self.previous_active, self.ever_selected = {}, {}, {}
        # Stage B: Goal query vectors cached by retrieval_goal_hash (upstream
        # _goal_query_cache) and the Goal of a turn between begin and plan.
        self.goal_cache, self.pending = {}, {}
        # Per session: the compact (projected, int8) form of every page vector, reused
        # across turns because a sealed page's vector never changes; and a counter of
        # catalog publications that receipts are planned against. Both replace work
        # that grew with the history (~4.5 ms per page per turn, and three digests of
        # the full catalog per turn, ~0.3 s at 68 pages).
        self.compact_cache, self.catalog_version = {}, {}
        self._projection = None
        vectors, schema, goals, selector = upstream()
        self.vectors, self.schema, self.selector = vectors, schema, selector
        # Decoupled layout (vector_format compact_int8_v1): Page vectors arrive
        # compacted by the worker (interface I3) and are never projected here.
        self.compact_vectors = selection_contract.vector_format(self.contract) == selection_contract.COMPACT_INT8_V1
        self.compact_type, self.provenance_type = memo_types(vectors)
        # Per session: one upstream index whose Pages are published once (a
        # sealed Page's vector never changes), like the upstream server's own
        # index; the per-turn plan publishes only new Pages and the query.
        self.page_index = {}
        # Vectorized selection (numpy module, or None for the reference path),
        # decided once here by fast_selection's self-check; per session the
        # Page unit vectors as one growing column matrix (a sealed Page's
        # vector never changes).
        self.fast = fast_selection.available(vectors)
        self.page_columns = {}
        self._config_index = None
        # Only goal_query starts the Goal rewriter. observation_query ranks Pages
        # by the observation vector alone: its Goal is the deterministic one of
        # --no-goal-rewriter (for_policy above already refused a pinned rewriter).
        pin = self.contract['goal_rewriter']
        require(rewriter_batch is None or (policy == 'goal_query' and pin is not None),
                '--goal-rewriter-batch needs goal_query with a pinned Goal rewriter')
        self.rewriter = ((goal_rewriter(pin) if rewriter_batch is None else goal_rewriter(pin, batch=rewriter_batch))
                         if selection_contract.uses_goal_rewriter(policy) else None)
        if rewriter_batch is not None:
            # Startup only: the rewriter runs batched with the pinned N.
            from .goal_rewriter_batch import check
            check(self.rewriter, rewriter_batch)
        self.rewriter_batch = rewriter_batch
        self.constructor = goals.MaintainedGoalConstructor(self.rewriter,
            entity_drift_steps=self.contract['entity_drift_steps'],
            initialize_with_rewriter=self.rewriter is not None)
        if transcript is None:
            # Tool-call protocol: reuse the reviewed extractor that never infers
            # exit/test verdicts from arbitrary tool stdout.  The XML protocol
            # keeps the upstream extractor and parses <returncode> like upstream.
            from goal_adapter import CapturedTextFeatureExtractor
            self.constructor.feature_extractor = CapturedTextFeatureExtractor()
        # goal_query evidence (goal_text, recording only): the texts behind each
        # turn's Goal hashes and its rewriter exchanges, kept from plan_turn to the
        # native terminal, which appends them to goals.jsonl. Startup only: the
        # tap leaves the control hash as it is, and the stored Goal text hashes
        # to the GoalRecord version hash.
        self.goal_texts = goal_text.GoalTexts()
        self.tap = goal_text.tap(self.constructor, selection_contract.constructor_control_hash)
        if policy == 'goal_query':
            goal_text.self_check(sha256, goals.MaintainedGoalConstructor, schema)
        self.goal_control_hash = selection_contract.constructor_control_hash(self.constructor)

    def close(self):
        if self.rewriter is not None:
            self.rewriter.close()

    def bound(self, session_id):
        """(trusted task, engine transcript) of one session."""
        if self.trusted is None:
            return self.task, self.transcript
        require(session_id in self.trusted, 'session not initialized by loaded server')
        return self.trusted[session_id]

    def tasks(self):
        """Every session's trusted task, in session order."""
        return [self.task] if self.trusted is None else [t for t, _ in self.trusted.values()]

    def catalog_token(self, binding):
        """The catalog publication a receipt was planned against (a counter, not a vector digest)."""
        width = self.adapter.page_tokens
        return '%s:%d:%d:%d' % (binding.session, binding.stable_end // width * width,
                                len(self.vector_catalog.get(binding.session, {})),
                                self.catalog_version.get(binding.session, 0))

    def projection(self, hidden_size):
        if self._projection is None:
            matrix, value = selection_contract.projection(hidden_size, self.contract)
            self._projection = (hidden_size, matrix, value)
        require(self._projection[0] == hidden_size, 'hidden size changed within one server')
        return self._projection[1], self._projection[2]

    def projection_sha256(self, hidden_size):
        """The vector space's projection hash without building the matrix when it is pinned."""
        pinned = selection_contract.PROJECTION_SHA256.get(hidden_size)
        if self._projection is None and pinned is not None:
            return pinned
        return self.projection(hidden_size)[1]

    def page_vector(self, value):
        """One Page vector in this server's representation: a CompactVector
        (compact_int8_v1, decoupled) or a normalised float row (legacy)."""
        if self.compact_vectors:
            return selection_contract.parse_compact(value, self.compact_type,
                                                    self.contract['projection_dimension'])
        return normalized(value)

    def query_vector(self, value):
        """The observation/Goal query: as the Pages in legacy mode; in compact mode a
        compact wire vector or a float row (compacted in plan, one per turn)."""
        if self.compact_vectors and (_compact_wire(value) or isinstance(value, self.compact_type)):
            return self.page_vector(value)
        return normalized(value)

    def same_space(self, entries, query):
        if not self.compact_vectors:
            return all(len(v) == len(query) for v in entries.values())
        dimension = self.contract['projection_dimension']
        return (all(type(v) is self.compact_type and v.dimension == dimension for v in entries.values())
                and (type(query) is not self.compact_type or query.dimension == dimension))

    def plan(self, binding, goal, context, entries, query, n, vector_space=None, local_turn=None):
        """Upstream ``_plan`` on native Pages: (ranking, selected count, evidence)."""
        contract = self.contract
        turn = binding.turn if local_turn is None else local_turn
        ends = self.turn_ends[binding.session]
        transcript = self.bound(binding.session)[1]
        initial = (transcript.initial_length if transcript is not None
                   else ends[0] if ends else binding.stable_end)
        mandatory = selection_contract.mandatory_pages(initial, n, contract) if n else 0
        evidence = dict(contract_sha256=self.contract_sha256, candidate_pages=n,
            history_tokens=selection_contract.history_tokens(n, contract), initial_prompt_end=initial)
        if not selection_contract.selection_active(n, turn, contract):
            evidence.update(reason='cold_start' if turn == 0 else 'below_switch_threshold',
                capacity_pages=n, selected_pages=list(range(n)), mandatory_pages=[],
                recent_pages=[], retrieved_pages=[], page_turns={}, projection_sha256=None,
                selector_config_hash=None, query_hash=digest(query))
            return tuple(range(n)), n, evidence
        # Upstream turn ids are 1-based.  The stable boundary seen at vLLM turn
        # k ends upstream turn k; the current stable prefix ends turn t.
        boundaries = ends[1:] + [binding.stable_end]
        capacity = selection_contract.capacity_pages(n, mandatory, contract)
        if self.adapter.sessions[binding.session].policy == selection_contract.RANDOM_FILL:
            return self.plan_random_fill(binding, goal, context, n, turn, mandatory, capacity,
                                         boundaries, evidence)
        if self.fast is not None and self.compact_vectors:
            planned = self.plan_fast(binding, entries, query, n, turn, mandatory, capacity,
                                     boundaries, evidence, vector_space)
            if planned is not None:
                return planned
        return self.plan_reference(binding, goal, context, entries, query, n, turn, mandatory,
                                   capacity, boundaries, evidence, vector_space)

    def plan_reference(self, binding, goal, context, entries, query, n, turn, mandatory, capacity,
                       boundaries, evidence, vector_space):
        """``plan`` once selection is active, through the upstream index and Selector V2."""
        contract, width = self.contract, self.adapter.page_tokens
        records, vectors, turns = [], {}, {}
        if self.compact_vectors:
            # The worker compacted every Page; only a float query is projected here.
            vectors = {page: entries[page] for page in range(n)}
            if type(query) is self.compact_type:
                hidden = (vector_space or {}).get('hidden_size')
                require(type(hidden) is int and hidden > 0, 'compact query needs the hidden size of its vector space')
                query_vector, projection_hash = query, self.projection_sha256(hidden)
            else:
                matrix, projection_hash = self.projection(len(query))
                query_vector = selection_contract.compact(query, matrix, self.compact_type)
        else:
            matrix, projection_hash = self.projection(len(query))
            compact = lambda values: selection_contract.compact(values, matrix, self.compact_type)
            cache = self.compact_cache.setdefault(binding.session, {})
            for page in range(n):
                hit = cache.get(page)
                if hit is None or hit[0] != entries[page]:
                    hit = cache[page] = (entries[page], compact(entries[page]))
                vectors[page] = hit[1]
            query_vector = compact(query)
        layer = (vector_space or {}).get('decoder_layers', 1) - 1
        provenance = self.provenance_type(binding.model, binding.renderer, max(0, layer),
            'last_token', projection_hash, contract['projection_dimension'], 'int8',
            vector_source='vllm_pre_final_norm_hidden_states')
        state = self.page_index.get(binding.session)
        if state is None or state[1] != provenance:
            index = self.vectors.PageVectorIndex()
            state = self.page_index[binding.session] = (
                index, provenance, self.vectors.PageVectorPublisher(index, provenance), {})
        index, provenance, publisher, refs = state
        for page in range(n):
            # One native Page is represented by its final 16-token upstream Page
            # (at P = 16, the native Page itself).
            start = (page+1)*width-16
            turns[page] = selection_contract.page_turn(page, boundaries, width)
            ref = refs.get(page)
            if ref is None or (ref[0] is not vectors[page] and ref[0] != vectors[page]):
                # An immutable Page: the upstream index refuses a different vector.
                ref = refs[page] = (vectors[page], publisher.publish_page(binding.session, str(page), vectors[page]))
            vectors[page] = ref[0]    # the published object (its derived values are kept)
            records.append(self.schema.PageRecord(page_id=str(page), turn_id=turns[page],
                role='history', token_start=start, token_end=start+16,
                logical_position_start=start, logical_position_end=start+16, byte_size=0,
                mandatory=page < mandatory, page_vector_ref=ref[1]))
        publisher.publish_query(binding.session, turn+1, query_vector)
        page_catalog = self.schema.PageCatalog(binding.session, tuple(records))
        selector_context = replace(context, historical_turn_ids=tuple(range(1, turn+1)),
            historical_page_ids=tuple(record.page_id for record in records))
        policy = self.adapter.sessions[binding.session].policy
        selector = self.selector.PrefillPageEmbeddingSelectorV2(index, provenance,
            frozen_goal_trace_hash=self.goal_control_hash, page_budget=capacity,
            recent_turns=contract['recent_turns'], recent_quota=contract['recent_quota_pages'],
            query_source='goal_hidden' if policy == 'goal_query' else 'observation_hidden')
        # Upstream previous_active is the resident set after the prior turn:
        # its selected Pages plus every Page sealed while serving it.
        selected, sealed = self.previous_active.get(binding.session, (frozenset(), n))
        previous = sorted(selected | set(range(sealed, n)))
        ever = sorted(p for p in self.ever_selected.get(binding.session, ()) if p < n)
        plan = selector.select(selector_context, goal, page_catalog,
            previous_active=tuple(map(str, previous)), ever_selected=tuple(map(str, ever)))
        require(plan.selector_config.get('effective_mode') == 'prefill_page_cosine',
                'Selector V2 fell back: %s' % plan.selector_config.get('fallback_reason'))
        scores = {page: self.vectors.compact_cosine(query_vector, vectors[page]) for page in range(n)}
        by_score = lambda pages: sorted(pages, key=lambda p: (-scores[p], p))
        mandatory_ids = [int(p) for p in plan.mandatory_page_ids]
        recent_ids = [int(p) for p in plan.recent_page_ids]
        retrieved_ids = by_score(int(p) for p in plan.retrieved_page_ids)
        chosen = set(mandatory_ids) | set(recent_ids) | set(retrieved_ids)
        require(chosen == {int(p) for p in plan.ordered_page_ids} and 0 < len(chosen) <= capacity,
                'Selector V2 plan exceeds its Page budget')
        ranking = tuple(mandatory_ids + recent_ids + retrieved_ids
                        + by_score(set(range(n)) - chosen))
        evidence.update(reason=None, capacity_pages=capacity, selected_pages=sorted(chosen),
            mandatory_pages=mandatory_ids, recent_pages=recent_ids, retrieved_pages=retrieved_ids,
            page_turns={str(p): turns[p] for p in range(n)}, projection_sha256=projection_hash,
            selector_config_hash=plan.selector_config['selector_config_hash'],
            added_pages=[int(p) for p in plan.added_page_ids],
            retained_pages=[int(p) for p in plan.retained_page_ids],
            removed_pages=[int(p) for p in plan.removed_page_ids],
            scores={str(p): round(scores[p], 8) for p in range(n)},
            query_hash=sha256(('%s:%s:%s' % (policy, provenance.provenance_hash,
                                              query_vector.content_hash)).encode()).hexdigest())
        return ranking, len(chosen), evidence

    # ---- vectorized selection (fast_selection) ------------------------------
    # The reference path above builds one upstream PageRecord per Page, publishes
    # the Pages to the upstream index, and the upstream selector scores every
    # Page in Python (and this module scores them again). Its outcome is fixed by
    # a few facts that hold for native Pages, which the fast path relies on and
    # checks (it returns None, i.e. the reference path runs, when one fails):
    # * every candidate Page belongs to an upstream turn <= t, so the selector's
    #   history (turn < t + 1) is the whole catalog [0, n), in Page order;
    # * a Page's logical position (p + 1) * P - 16 grows with p, so the
    #   tie-breaks (-cosine, position, page id) and (-turn, -position, page id)
    #   are (-cosine, p) and (-turn, -p);
    # * the selector's budget is in Pages (page_budget = capacity), so every
    #   Page costs 1 and the greedy loops take prefixes of their orders.
    # The cosine floats are the upstream ones (fast_selection.cosines); only the
    # evidence fields are produced, with the same values, types and key order.

    def _partition(self, np, n, turn, mandatory, capacity, boundaries):
        """(Page turns, recent Pages ascending) of the upstream ``_safety_partition``, or None."""
        width, contract = self.adapter.page_tokens, self.contract
        ends = np.asarray(boundaries, dtype=np.int64)
        if n <= 0 or not 0 <= mandatory <= min(n, capacity) or (ends.size > 1 and not (np.diff(ends) > 0).all()):
            return None
        pages = np.arange(n, dtype=np.int64)
        # selection_contract.page_turn: 1 + bisect_right(boundaries, last token)
        turns = np.searchsorted(ends, (pages + 1) * width - 1, side='right') + 1
        if int(turns.max()) > turn:
            return None
        cutoff = max(0, turn + 1 - contract['recent_turns'])
        eligible = pages[mandatory:][turns[mandatory:] >= cutoff]
        eligible = eligible[np.lexsort((-eligible, -turns[eligible]))]
        take = min(contract['recent_quota_pages'], max(0, capacity - mandatory), len(eligible))
        return turns, np.sort(eligible[:take])

    def _delta(self, np, binding, n, chosen):
        """(added, retained, removed) of upstream ``active_set_delta`` against the
        resident set after the prior turn (its selected Pages plus every Page
        sealed while serving it), as ints in Page order; None if that set
        reaches past n (the reference path then decides)."""
        selected, sealed = self.previous_active.get(binding.session, (frozenset(), n))
        previous = np.zeros(n, dtype=np.bool_)
        if selected:
            old = np.fromiter(selected, dtype=np.int64, count=len(selected))
            if int(old.min()) < 0 or int(old.max()) >= n:
                return None
            previous[old] = True
        if sealed < n:
            previous[max(0, sealed):] = True
        current = np.flatnonzero(chosen)
        before = np.flatnonzero(previous)
        return (current[~previous[current]].tolist(), current[previous[current]].tolist(),
                before[~chosen[before]].tolist())

    def _selector_config_hash(self, selector, extra):
        return selector._complete_config(extra)['selector_config_hash']

    def plan_fast(self, binding, entries, query, n, turn, mandatory, capacity, boundaries, evidence,
                  vector_space):
        """``plan_reference``'s ranking, count and evidence, vectorized (compact_int8_v1)."""
        np, contract = self.fast, self.contract
        if type(query) is self.compact_type:
            hidden = (vector_space or {}).get('hidden_size')
            require(type(hidden) is int and hidden > 0, 'compact query needs the hidden size of its vector space')
            query_vector, projection_hash = query, self.projection_sha256(hidden)
        else:
            matrix, projection_hash = self.projection(len(query))
            query_vector = selection_contract.compact(query, matrix, self.compact_type)
        layer = (vector_space or {}).get('decoder_layers', 1) - 1
        provenance = self.provenance_type(binding.model, binding.renderer, max(0, layer),
            'last_token', projection_hash, contract['projection_dimension'], 'int8',
            vector_source='vllm_pre_final_norm_hidden_states')
        partition = self._partition(np, n, turn, mandatory, capacity, boundaries)
        if partition is None:
            return None
        turns, recent = partition
        columns = self.page_columns.get(binding.session)
        if columns is None or columns.key != provenance:
            columns = self.page_columns[binding.session] = fast_selection.PageColumns(
                np, provenance, contract['projection_dimension'], self.vectors.CompactVector.unit_values)
        scores = fast_selection.cosines(np, columns.view(entries, n), query_vector.unit_values())
        chosen = np.zeros(n, dtype=np.bool_)
        chosen[:mandatory] = True
        chosen[recent] = True
        candidates = np.flatnonzero(~chosen)
        # sorted by (-cosine, Page): upstream (-score, logical position, page id)
        order = candidates[np.lexsort((candidates, -scores[candidates]))]
        retrieved = order[:min(len(order), max(0, capacity - mandatory - len(recent)))]
        chosen[retrieved] = True
        count = int(chosen.sum())
        require(0 < count <= capacity, 'Selector V2 plan exceeds its Page budget')
        delta = self._delta(np, binding, n, chosen)
        if delta is None:
            return None
        policy = self.adapter.sessions[binding.session].policy
        if self._config_index is None:
            self._config_index = self.vectors.PageVectorIndex()
        selector = self.selector.PrefillPageEmbeddingSelectorV2(self._config_index, provenance,
            frozen_goal_trace_hash=self.goal_control_hash, page_budget=capacity,
            recent_turns=contract['recent_turns'], recent_quota=contract['recent_quota_pages'],
            query_source='goal_hidden' if policy == 'goal_query' else 'observation_hidden')
        recent_ids, retrieved_ids, rest = recent.tolist(), retrieved.tolist(), order.tolist()
        keys = page_keys(n)
        evidence.update(reason=None, capacity_pages=capacity, selected_pages=np.flatnonzero(chosen).tolist(),
            mandatory_pages=list(range(mandatory)), recent_pages=recent_ids, retrieved_pages=retrieved_ids,
            page_turns=dict(zip(keys, turns.tolist())), projection_sha256=projection_hash,
            selector_config_hash=self._selector_config_hash(selector, {
                'effective_mode': 'prefill_page_cosine',
                'stable_tie_break': 'semantic_rank,logical_position,page_id'}),
            added_pages=delta[0], retained_pages=delta[1], removed_pages=delta[2],
            scores=dict(zip(keys, map(round, scores.tolist(), repeat(8, n)))),
            query_hash=sha256(('%s:%s:%s' % (policy, provenance.provenance_hash,
                                              query_vector.content_hash)).encode()).hexdigest())
        return tuple(range(mandatory)) + tuple(recent_ids) + tuple(rest), count, evidence

    def plan_random_fill_fast(self, binding, n, turn, mandatory, capacity, boundaries, evidence):
        """``plan_random_fill_reference``'s ranking, count and evidence, vectorized."""
        np, contract = self.fast, self.contract
        partition = self._partition(np, n, turn, mandatory, capacity, boundaries)
        if partition is None:
            return None
        turns, recent = partition
        fill = contract[selection_contract.RANDOM_FILL]
        session_index = self.session_index[binding.session]
        chosen = np.zeros(n, dtype=np.bool_)
        chosen[:mandatory] = True
        chosen[recent] = True
        order = fast_selection.fill_order(np, np.flatnonzero(~chosen).tolist(), fill['seed'],
                                          session_index, binding.turn, selection_contract.fill_order)
        retrieved_ids = order[:min(len(order), max(0, capacity - mandatory - len(recent)))]
        chosen[retrieved_ids] = True
        count = int(chosen.sum())
        require(0 < count <= capacity, 'random fill plan exceeds its Page budget')
        recent_ids = recent.tolist()
        ranking = tuple(range(mandatory)) + tuple(recent_ids) + tuple(order)
        require(len(ranking) == n, 'random fill ranking must be a Page permutation')
        delta = self._delta(np, binding, n, chosen)
        if delta is None:
            return None
        selector = random_fill_selector(self.selector)(frozen_goal_trace_hash=self.goal_control_hash,
            page_budget=capacity, recent_turns=contract['recent_turns'],
            recent_quota=contract['recent_quota_pages'], seed=fill['seed'],
            session_index=session_index, turn=binding.turn)
        evidence.update(reason=None, capacity_pages=capacity, selected_pages=np.flatnonzero(chosen).tolist(),
            mandatory_pages=list(range(mandatory)), recent_pages=recent_ids, retrieved_pages=retrieved_ids,
            page_turns=dict(zip(page_keys(n), turns.tolist())), projection_sha256=None,
            selector_config_hash=self._selector_config_hash(selector, {
                'effective_mode': 'random_fill', 'stable_tie_break': 'none'}),
            added_pages=delta[0], retained_pages=delta[1], removed_pages=delta[2],
            fill=dict(rng=fill['rng'], seed=fill['seed'], session_index=session_index,
                      turn=binding.turn, eligible_pages=len(order), drawn_pages=len(retrieved_ids)),
            query_hash=None)
        return ranking, count, evidence

    def plan_random_fill(self, binding, goal, context, n, turn, mandatory, capacity, boundaries,
                         evidence):
        if self.fast is not None:
            planned = self.plan_random_fill_fast(binding, n, turn, mandatory, capacity, boundaries,
                                                 evidence)
            if planned is not None:
                return planned
        return self.plan_random_fill_reference(binding, goal, context, n, turn, mandatory, capacity,
                                               boundaries, evidence)

    def plan_random_fill_reference(self, binding, goal, context, n, turn, mandatory, capacity,
                                   boundaries, evidence):
        """``plan`` of random_fill once selection is active: the Page records, the
        upstream safety partition and budget loop of Selector V2, the fill drawn at
        random. No query, projection, score or Page vector index: the records
        carry no vector reference. The draw is keyed by the pinned seed, the
        session index and the session's request ordinal (``binding.turn``)."""
        contract, width = self.contract, self.adapter.page_tokens
        records, turns = [], {}
        for page in range(n):
            start = (page+1)*width-16
            turns[page] = selection_contract.page_turn(page, boundaries, width)
            records.append(self.schema.PageRecord(page_id=str(page), turn_id=turns[page],
                role='history', token_start=start, token_end=start+16,
                logical_position_start=start, logical_position_end=start+16, byte_size=0,
                mandatory=page < mandatory, page_vector_ref=None))
        page_catalog = self.schema.PageCatalog(binding.session, tuple(records))
        selector_context = replace(context, historical_turn_ids=tuple(range(1, turn+1)),
            historical_page_ids=tuple(record.page_id for record in records))
        fill = contract[selection_contract.RANDOM_FILL]
        session_index = self.session_index[binding.session]
        selector = random_fill_selector(self.selector)(frozen_goal_trace_hash=self.goal_control_hash,
            page_budget=capacity, recent_turns=contract['recent_turns'],
            recent_quota=contract['recent_quota_pages'], seed=fill['seed'],
            session_index=session_index, turn=binding.turn)
        selected, sealed = self.previous_active.get(binding.session, (frozenset(), n))
        previous = sorted(selected | set(range(sealed, n)))
        ever = sorted(p for p in self.ever_selected.get(binding.session, ()) if p < n)
        plan = selector.select(selector_context, goal, page_catalog,
            previous_active=tuple(map(str, previous)), ever_selected=tuple(map(str, ever)))
        require(plan.selector_config.get('effective_mode') == 'random_fill', 'random fill selector changed')
        mandatory_ids = [int(p) for p in plan.mandatory_page_ids]
        recent_ids = [int(p) for p in plan.recent_page_ids]
        order = [int(p) for p in selector.order]
        retrieved_ids = order[:len(plan.retrieved_page_ids)]
        chosen = set(mandatory_ids) | set(recent_ids) | set(retrieved_ids)
        require(set(retrieved_ids) == {int(p) for p in plan.retrieved_page_ids}
                and chosen == {int(p) for p in plan.ordered_page_ids} and 0 < len(chosen) <= capacity,
                'random fill plan exceeds its Page budget')
        ranking = tuple(mandatory_ids + recent_ids + order)
        require(sorted(ranking) == list(range(n)), 'random fill ranking must be a Page permutation')
        evidence.update(reason=None, capacity_pages=capacity, selected_pages=sorted(chosen),
            mandatory_pages=mandatory_ids, recent_pages=recent_ids, retrieved_pages=retrieved_ids,
            page_turns={str(p): turns[p] for p in range(n)}, projection_sha256=None,
            selector_config_hash=plan.selector_config['selector_config_hash'],
            added_pages=[int(p) for p in plan.added_page_ids],
            retained_pages=[int(p) for p in plan.retained_page_ids],
            removed_pages=[int(p) for p in plan.removed_page_ids],
            fill=dict(rng=fill['rng'], seed=fill['seed'], session_index=session_index,
                      turn=binding.turn, eligible_pages=len(order), drawn_pages=len(retrieved_ids)),
            query_hash=None)
        return ranking, len(chosen), evidence

    # ---- Stage B: two-phase turns (XML protocol) -------------------------
    # begin_turn runs before phase 1 (the Goal needs only the observation text);
    # plan_turn consumes the phase-1 vectors; publish_completion commits every
    # vector at the terminal; reset_session implements an upstream rebuild.
    # Multi-session stage 2: a goal_query Goal may be built off the engine
    # thread between begin_turn's context and its remainder (goal_async).

    def _drop_tables(self, session_id):
        for table in (self.vector_catalog, self.turn_ends, self.previous_active,
                      self.ever_selected, self.goal_cache, self.compact_cache, self.catalog_version,
                      self.page_index, self.__dict__.get('page_columns', {})):
            table.pop(session_id, None)

    def reset_session(self, session_id):
        """Upstream rebuild: a fresh session with new Goal, index and selector state."""
        self._drop_tables(session_id)
        self.constructor.release_session(session_id)

    def close_session(self, session_id, goals=None):
        """Session pool: the session closed for good. Its tables go as on a
        rebuild; its Goal state is released on the Goal thread when one exists
        (it may still be building a discarded Goal of this session, and the
        engine thread must not wait for the rewriter), else here. Returns where."""
        self._drop_tables(session_id)
        if goals is not None and goals.after_goals(self.constructor.release_session, session_id):
            return 'goal_thread'
        self.constructor.release_session(session_id)
        return 'inline'

    def active(self, geometry):
        return selection_contract.selection_active(geometry.n, geometry.local_turn, self.contract)

    def begin_turn(self, binding, geometry, conversation):
        """Construct the Goal once per turn; say whether a Goal forward is needed."""
        from .goal_async import run_inline
        return run_inline(self.begin_turn_steps(binding, geometry, conversation))

    def begin_turn_steps(self, binding, geometry, conversation):
        """``begin_turn`` as an admission generator (goal_async). A Goal that may
        call the rewriter (goal_query) is yielded as one GoalWork with the context
        built here; the driver returns the GoalRecord, built in place or, on a
        server that parks goal_query admissions, on its Goal thread."""
        from . import mswea_protocol
        from . import kv_session_protocol
        session = self.adapter.sessions[binding.session]
        task, transcript = self.bound(binding.session)
        require(session.policy in ('goal_query', 'observation_query', 'random_fill') and transcript is not None,
                'Stage B selection serves query policies of the XML protocol')
        require(session.policy == self.policy, 'session policy differs from the selection authority policy')
        require(binding.request not in self.authorized and binding.request not in self.used
                and binding.request not in self.pending, 'query turn replay')
        # Spec 0/3.10: a local turn 0 (start or rebuild) has no candidates (n = 0)
        # whatever its stable end; later turns have n = floor(o/P).
        require(geometry.page_tokens == self.adapter.page_tokens and geometry.o == binding.stable_end
                and geometry.n == (binding.stable_end // geometry.page_tokens
                                   if geometry.local_turn else 0), 'turn geometry mismatch')
        ends = self.turn_ends.setdefault(binding.session, [])
        require(len(ends) == geometry.local_turn and all(e < binding.stable_end for e in ends),
                'stable turn boundaries must grow once per local turn')
        from . import chat_text_protocol
        if kv_session_protocol.kv(transcript):
            context = self.kv_context(binding, geometry, conversation, task, transcript)
        elif chat_text_protocol.chat(transcript):
            context = self.chat_context(binding, geometry, conversation, task, transcript)
        else:
            context = self.xml_context(binding, geometry, conversation, task, transcript)
        rewrites = ()
        if self.rewriter is None:
            goal = self.constructor.construct(context)
        else:
            from .goal_async import GoalWork
            work = GoalWork(binding.request, self.constructor, context, tap=self.tap)
            goal = yield work
            rewrites = work.rewrites
        variant = self.contract['goal_query_variant']
        text = selection_contract.goal_query_text(goal, variant)
        text_sha256 = sha256(text.encode()).hexdigest()
        cached = None
        if session.policy == 'goal_query' and self.active(geometry):
            # Upstream _goal_query_cache: keyed by retrieval_goal_hash; a hash
            # hit with a different rendered payload is a collision.
            cached = self.goal_cache.get(binding.session, {}).get(goal.retrieval_goal_hash)
            require(cached is None or cached[2] == text_sha256, 'Goal hash collision changed the query payload')
        need = session.policy == 'goal_query' and self.active(geometry) and cached is None
        query = dict(goal=goal, context=context, variant=variant, text=text, text_sha256=text_sha256,
                     goal_payload=text if need else None, cache_hit=cached is not None, geometry=geometry)
        self.pending[binding.request] = dict(query, rewrites=rewrites)
        return types.SimpleNamespace(**query)

    def xml_context(self, binding, geometry, conversation, task, transcript):
        """Mini-SWE-Agent XML TurnContext (upstream _context)."""
        from . import mswea_protocol
        require(type(conversation) is list and conversation
                and all(type(m) is dict and type(m.get('content')) is str for m in conversation)
                and conversation[-1].get('role') == 'user', 'server-rendered XML conversation required')
        require(transcript.pending is not None
                and transcript.turn == geometry.local_turn, 'engine transcript is not at this turn')
        if geometry.local_turn == 0:
            require(any(m.get('role') == 'user' and task['problem_statement'] in m['content']
                        for m in conversation), 'trusted issue absent from initial conversation')
        else:
            require(binding.observation_span is not None, 'latest observation span required')
        # Upstream _context: the latest user message (the task on the first
        # turn), this session's previous decoded reply, and its <returncode>.
        observation = conversation[-1]['content']
        objective = mswea_protocol.root_objective(conversation)
        require(objective and task['problem_statement'].strip() in conversation[1]['content'],
                'task message must carry the frozen issue')
        return self.schema.TurnContext(binding.session, task['instance_id'],
            geometry.local_turn + 1, objective, None, transcript.previous_action(), observation,
            message_source='user' if geometry.local_turn == 0 else 'tool',
            structured_state={'tool_name': 'bash', 'return_code': mswea_protocol.return_code(observation)})

    def chat_context(self, binding, geometry, conversation, task, transcript):
        """chat-text TurnContext (the deterministic Goal is evidence only: chat-text serves
        observation_query and random_fill, which rank without it): the first user message
        as the root objective, the previous decoded reply, the newest user message as the
        observation, no tool name or return code."""
        from . import chat_text_protocol as chat
        require(type(conversation) is list and conversation
                and all(type(m) is dict and type(m.get('content')) is str for m in conversation)
                and conversation[-1].get('role') == 'user', 'server-rendered chat-text conversation required')
        require(transcript.pending is not None and transcript.turn == geometry.local_turn,
                'engine transcript is not at this turn')
        first = geometry.local_turn == 0
        if not first:
            require(binding.observation_span is not None, 'latest observation span required')
        fields = chat.goal_inputs(conversation, first)
        return self.schema.TurnContext(binding.session, task['instance_id'], geometry.local_turn + 1,
            fields['objective'], None, transcript.previous_action(), fields['observation'],
            message_source=fields['message_source'], structured_state=fields['structured_state'])

    def kv_context(self, binding, geometry, source, task, transcript):
        """kv-session TurnContext: the root objective (the first user message), this
        session's previous decoded reply, and the new messages: their tool/user text as
        the observation, a real user message (the MemAgent question, a format-error
        nudge) as the explicit requirement, the reply's tool name and <returncode>."""
        from . import kv_session_protocol as kv
        require(type(source) is dict and source.get('schema') == kv.SOURCE_SCHEMA,
                'kv-session source required')
        require(transcript.pending is not None and transcript.turn == geometry.local_turn,
                'engine transcript is not at this turn')
        first = geometry.local_turn == 0
        if not first:
            require(binding.observation_span is not None, 'latest observation span required')
        objective = source['objective'] if first else transcript.objective
        require(type(objective) is str, 'kv-session root objective missing')
        fields = kv.goal_inputs(source['messages'], first)
        return self.schema.TurnContext(binding.session, task['instance_id'], geometry.local_turn + 1, objective,
            None, transcript.previous_action(), fields['observation'], message_source=fields['message_source'],
            structured_state=fields['structured_state'], explicit_requirement=fields['explicit_requirement'])

    def plan_turn(self, binding, geometry, allocation_generation, query, phase1):
        """Selector V2 over candidates [0, n); returns (ranking, count, signed receipt)."""
        pending = self.pending.pop(binding.request, None)
        require(pending is not None and pending['goal'] is query.goal and pending['geometry'] == geometry,
                'plan_turn without its begin_turn')
        session = self.adapter.sessions[binding.session]
        width, n = self.adapter.page_tokens, geometry.n
        catalog = self.vector_catalog.setdefault(binding.session, {})
        require(set(catalog) == set(range(geometry.Q // width)), 'committed vector catalog mismatch')
        require((phase1 is None) == (geometry.local_turn == 0), 'phase 1 runs on every turn after the first')
        entries = {page: vector for page, vector in catalog.items() if page < n}
        if phase1 is not None:
            require(phase1.binding == binding.identity and phase1.start == geometry.C
                    and phase1.end == geometry.e, 'foreign phase-1 result')
            fresh = {int(page): self.page_vector(vector) for page, vector in phase1.page_vectors.items()}
            require(set(fresh) == {x // width for x in range(geometry.C, geometry.e) if (x + 1) % width == 0}
                    and not set(fresh) & set(catalog), 'phase 1 must publish exactly the pages it closed')
            entries.update({page: vector for page, vector in fresh.items() if page < n})
        # A cold (local turn 0) request computes its pages' first vectors only
        # now; selection is inactive there and needs none.
        require(set(entries) == set(range(n)) or geometry.local_turn == 0, 'candidate vectors incomplete')
        policy = session.policy
        query_vector, goal_forward = None, False
        if policy == selection_contract.RANDOM_FILL and self.active(geometry):
            # No query vector and no Goal forward: the fill is drawn at random.
            require(phase1.goal_vector is None, 'random_fill runs no Goal forward')
            ranking, count, selection = self.plan(binding, query.goal, query.context, entries, None,
                n, None, geometry.local_turn)
        elif self.active(geometry):
            if policy == 'goal_query':
                cache = self.goal_cache.setdefault(binding.session, {})
                key = query.goal.retrieval_goal_hash
                if query.cache_hit:
                    require(phase1.goal_vector is None, 'cached Goal must not run a Goal forward')
                    query_vector = cache[key][0]
                else:
                    require(phase1.goal_vector is not None
                            and phase1.goal_payload_sha256 == query.text_sha256, 'Goal forward missing or foreign')
                    query_vector, goal_forward = self.query_vector(phase1.goal_vector), True
                    cache[key] = (query_vector, phase1.goal_token_count, query.text_sha256)
            else:
                query_vector = self.query_vector(phase1.observation_vector)
            require(self.same_space(entries, query_vector), 'query/Page space mismatch')
            ranking, count, selection = self.plan(binding, query.goal, query.context, entries, query_vector,
                n, phase1.vector_space if phase1 is not None else None, geometry.local_turn)
        else:
            ranking, count, selection = self.plan(binding, query.goal, query.context, entries,
                (1.0,), n, None, geometry.local_turn)
        body = dict(binding=binding.identity, session=binding.session, turn=binding.turn,
            local_turn=geometry.local_turn, epoch=binding.epoch, model=binding.model,
            renderer=binding.renderer, generation=binding.generation,
            allocation_generation=allocation_generation, candidate_pages=n,
            catalog_version=self.catalog_token(binding), ranking=list(ranking), selected_count=count,
            selection=selection, query_hash=selection['query_hash'], goal_hash=query.goal.version_hash,
            goal=dict(variant=query.variant, payload_sha256=query.text_sha256,
                gate_decision=query.goal.gate_decision, applied_decision=query.goal.applied_decision,
                slm_attempted=query.goal.slm_attempted, slm_succeeded=query.goal.slm_succeeded,
                fallback_reason=query.goal.fallback_reason, cache_hit=query.cache_hit,
                forward=goal_forward),
            phase1=None if phase1 is None else dict(lease_id=phase1.lease_id, lease_digest=phase1.lease_digest,
                start=phase1.start, end=phase1.end, state_digest=phase1.state_digest,
                tail_capture_digest=phase1.tail_capture_digest),
            nonce=secrets.token_hex(16))
        receipt = dict(body=body, mac=self.adapter.authority.sign(body))
        if policy == 'goal_query':
            # Evidence only (goal_text): the texts of the two hashes above, until
            # the native terminal appends them to goals.jsonl.
            self.goal_texts.pending[binding.request] = goal_text.record(query.goal, query.variant,
                query.text, body['goal_hash'], query.text_sha256, pending.get('rewrites', ()))
        self.authorized[binding.request] = receipt
        self.turn_ends[binding.session].append(binding.stable_end)
        self.previous_active[binding.session] = (frozenset(ranking[:count]), n)
        self.ever_selected.setdefault(binding.session, set()).update(ranking[:count])
        return ranking, count, receipt

    def publish_completion(self, binding, geometry, phase1_vectors, completion_vectors,
                           sealed_end, computed_end):
        """Commit the first-computation vector of every page sealed by this turn."""
        width = self.adapter.page_tokens
        catalog = self.vector_catalog.setdefault(binding.session, {})
        values = {int(page): self.page_vector(vector) for page, vector in phase1_vectors.items()}
        later = {int(page): self.page_vector(vector) for page, vector in completion_vectors.items()}
        require(not set(values) & set(later) and not (set(values) | set(later)) & set(catalog)
                and sealed_end % width == 0 and sealed_end <= computed_end, 'stale or duplicate vector publication')
        catalog.update(values); catalog.update(later)
        require(set(catalog) == set(range(sealed_end // width)), 'published vectors differ from sealed pages')
        self.catalog_version[binding.session] = self.catalog_version.get(binding.session, 0) + 1

    def verify_reservation(self,binding,ranking,selected_count,receipt):
        require(type(receipt) is dict and set(receipt)=={'body','mac'}, 'query receipt required')
        self.adapter.authority.verify(receipt['body'],receipt['mac'])
        require(binding.request not in self.used and self.authorized.get(binding.request)==receipt,
                'query receipt replay or foreign issuer')
        body=receipt['body']
        require(body['binding']==binding.identity and tuple(body['ranking'])==tuple(ranking)
                and body['selected_count']==selected_count
                and body['catalog_version']==self.catalog_token(binding),
                'stale catalog or tampered query reservation')
        self.used.add(binding.request)
