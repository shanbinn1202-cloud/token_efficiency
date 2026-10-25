"""Online Selector V2 contract of dram-backed-kv, adapted to native P-token Pages.

Upstream reference: dram-backed-kv ``scripts/serve_online_selector_v2.py``
(main a70701e; behaviour-identical to the delta baseline ca0ce27 for Goal,
rewriter and selector).  Only the Page granularity differs: vLLM restores
full-attention KV in P-token Pages (784 with the legacy shared pool), so
upstream token budgets are converted to whole Pages (floored) and every native
Page is represented by its final 16-token upstream Page.  Everything else keeps
the upstream values and algorithms:

* no selection until the sealed history exceeds ``switch_threshold_tokens``;
* capacity ``max(1 Page, mandatory Pages, floor(history * active_ratio))``;
* the initial system + task prompt is mandatory (every Page it touches);
* the two previous turns are a recent safety set, bounded by
  ``recent_quota_pages`` (upstream 64 x 16 tokens = 1,024 tokens ~ 2 Pages);
* 128-d seeded Rademacher projection of the pre-final-norm last-token hidden
  state, L2 normalised and int8 symmetric per-vector quantised;
* ``PrefillPageEmbeddingSelectorV2`` ranks the remaining Pages by cosine.

Decoupled KV layout (FA Pages of P tokens, P in {16, 32, 64, 784}, a separate
GDN pool): the Page size is a contract parameter, so at P = 16 the contract is
the upstream one (recent quota 64 Pages, budgets floored to 16 tokens, one
vector per native Page) and the "final 16-token upstream Page" of a native Page
is the Page itself.  The worker compacts the closing rows on the device and
sends int8 vectors (``vector_format`` ``compact_int8_v1``, wire form
``[[q_0, ..., q_127], scale]`` per Page); the engine never projects Page rows.
Only such a contract carries the ``vector_format`` key: the legacy contract is
byte-identical to the 784-token one above, and so is its digest.

Random-fill control (policy ``random_fill``): the same threshold, capacity,
mandatory and recent Pages as the query policies, but the remaining capacity is
filled with candidate Pages drawn uniformly at random without replacement
instead of the highest-cosine ones.  Only its contract carries the
``random_fill`` key ({seed, rng}); the draw is ``fill_order`` (SplitMix64 and
Fisher-Yates, keyed by the pinned seed, the session index and the session's
request ordinal: integer arithmetic, nothing hashed).  Every other contract
keeps its bytes and its digest.
"""
from __future__ import annotations

from dataclasses import asdict
from bisect import bisect_right
from hashlib import sha256
import json
import math
from pathlib import Path

from .contract import require

SCHEMA = 'remote-ra-selection-contract-v1'
UPSTREAM = 'dram-backed-kv scripts/serve_online_selector_v2.py a70701e'
UPSTREAM_PAGE_TOKENS = 16
UPSTREAM_RECENT_QUOTA = 64
PROJECTION_DIMENSION = 128
PROJECTION_SEED = 20260904
# sha256 of the float32 projection for hidden size 5120, dimension 128 and the
# seed above; equals the projection_hash recorded by the upstream cohort-2 runs.
# 5376 is Gemma 4 31B IT's hidden size (gemma-s1, the same builder and seed; D-9).
PROJECTION_SHA256 = {5120: '20c2e1cf4463ed9407a9f9af0a7794fe044c02c0da369128681270a7427c87aa',
                     5376: '852b42e43dc0beb2d2ea90deefc549a36acc0e08c088624bc0a4588658a1dfb7'}
GOAL_QUERY_VARIANTS = ('full_goal', 'candidate_entity_cap8')
REWRITER_KEYS = {'model', 'config_sha256', 'max_new_tokens', 'timeout_seconds',
                 'startup_timeout_seconds'}
KEYS = {'schema', 'upstream', 'page_tokens', 'switch_threshold_tokens', 'active_ratio',
        'active_token_budget', 'recent_turns', 'recent_quota_pages', 'mandatory',
        'projection_dimension', 'projection_seed', 'vector_dtype', 'hidden_state',
        'goal_query_variant', 'entity_drift_steps', 'goal_rewriter'}
# Worker-compacted int8 Page vectors (decoupled layout, interface I3). The legacy
# representation (normalised float rows, compacted by the engine) has no key.
COMPACT_INT8_V1 = 'compact_int8_v1'
LEGACY_REPRESENTATION = 'pre_final_norm_last_token_l2'
VECTOR_FORMATS = (COMPACT_INT8_V1,)
# The random-fill control: its contract alone carries this key.
RANDOM_FILL = 'random_fill'
RANDOM_FILL_RNG = 'splitmix64_fisher_yates_v1'
RANDOM_FILL_KEYS = {'seed', 'rng'}
DEFAULT_FILL_SEED = 42
FILL_SEED_LIMIT = 1 << 32
OPTIONAL_KEYS = {'vector_format', RANDOM_FILL}


def default_recent_quota_pages(page_tokens):
    return math.ceil(UPSTREAM_RECENT_QUOTA * UPSTREAM_PAGE_TOKENS / page_tokens)


def default_contract(page_tokens=784, *, active_ratio=0.40, active_token_budget=None,
                     goal_rewriter=None, goal_query_variant='full_goal', vector_format=None,
                     random_fill=None):
    """``vector_format`` None: the legacy contract (no key); ``compact_int8_v1``: a
    decoupled-layout server whose worker sends compact Page vectors
    (``vector_format_for(kv_layout)``). ``random_fill`` None: no key (every policy
    but random_fill); else the random-fill pin (``random_fill_pin(seed)``)."""
    contract = dict(schema=SCHEMA, upstream=UPSTREAM, page_tokens=page_tokens,
        switch_threshold_tokens=8192, active_ratio=active_ratio,
        active_token_budget=active_token_budget, recent_turns=2,
        recent_quota_pages=default_recent_quota_pages(page_tokens),
        mandatory='initial_prompt_pages', projection_dimension=PROJECTION_DIMENSION,
        projection_seed=PROJECTION_SEED, vector_dtype='int8', hidden_state='pre_final_norm',
        goal_query_variant=goal_query_variant, entity_drift_steps=3,
        goal_rewriter=goal_rewriter)
    if vector_format is not None:
        contract['vector_format'] = vector_format
    if random_fill is not None:
        contract[RANDOM_FILL] = random_fill
    return validate(contract)


def random_fill_pin(seed=DEFAULT_FILL_SEED):
    """The ``random_fill`` contract value: the startup-pinned seed and the draw."""
    return validate_random_fill(dict(seed=seed, rng=RANDOM_FILL_RNG))


def validate_random_fill(value):
    require(type(value) is dict and set(value) == RANDOM_FILL_KEYS and value['rng'] == RANDOM_FILL_RNG
            and type(value['seed']) is int and 0 <= value['seed'] < FILL_SEED_LIMIT,
            'random fill pin {seed: int in [0, 2^32), rng: %r} required' % RANDOM_FILL_RNG)
    return value


def vector_format(contract):
    """None (legacy float rows) or ``compact_int8_v1``."""
    return contract.get('vector_format')


LAYOUT_KEY = 'remote_ra_kv_layout'   # kv_layout.LAYOUT_KEY (interface I0)


def layout_fields(kv_layout):
    """{page_tokens, mamba_tokens, mamba_slots} of a decoupled layout record, None for legacy.

    Accepts the startup config's ``kv_layout`` (that inner dict, as serve.py
    writes it), the whole ``additional_config`` ({remote_ra_kv_layout: {...}},
    ``kv_layout.to_additional_config``) or the KvLayout fields (with
    ``decoupled``); None or absent is the legacy shared pool."""
    if kv_layout is None:
        return None
    value = kv_layout.get(LAYOUT_KEY, kv_layout) if type(kv_layout) is dict else None
    # The window layout of a sliding-window model (gemma-s1) names its state instead.
    require(type(value) is dict and ({'page_tokens', 'mamba_tokens', 'mamba_slots'} <= set(value)
                                      or value.get('state') == 'sliding' and 'page_tokens' in value)
            and value.get('decoupled', True) is True, 'kv_layout record must describe a decoupled layout')
    return value


def layout_decoupled(kv_layout):
    """The startup ``kv_layout`` record (interface I0) names a decoupled layout."""
    return layout_fields(kv_layout) is not None


def vector_format_for(kv_layout):
    """The vector format a server of this KV layout runs (decoupled -> compact)."""
    return COMPACT_INT8_V1 if layout_decoupled(kv_layout) else None


def check_layout(contract, kv_layout, page_tokens=None):
    """Startup check: the contract's Page size and vector format match the KV layout."""
    validate(contract)
    fields = layout_fields(kv_layout)
    require(vector_format(contract) == vector_format_for(kv_layout),
            'selection contract vector format %r differs from the KV layout (%s)'
            % (vector_format(contract), 'decoupled' if fields is not None else 'legacy'))
    if fields is not None:
        require(contract['page_tokens'] == fields['page_tokens'],
                'selection contract Page size differs from the KV layout')
    require(page_tokens is None or contract['page_tokens'] == page_tokens,
            'selection contract Page size differs from native Pages')
    return contract


def validate(contract):
    require(type(contract) is dict and set(contract) - OPTIONAL_KEYS == KEYS
            and contract['schema'] == SCHEMA and contract['upstream'] == UPSTREAM,
            'exact selection contract required')
    require(contract.get('vector_format', COMPACT_INT8_V1) in VECTOR_FORMATS,
            'unknown selection vector format')
    if RANDOM_FILL in contract:
        validate_random_fill(contract[RANDOM_FILL])
    page = contract['page_tokens']
    require(type(page) is int and page > 0 and page % UPSTREAM_PAGE_TOKENS == 0,
            'selection Pages must be 16-token aligned')
    require(type(contract['switch_threshold_tokens']) is int and contract['switch_threshold_tokens'] >= 0,
            'switch threshold must be a non-negative token count')
    ratio, budget = contract['active_ratio'], contract['active_token_budget']
    require((ratio is None) != (budget is None), 'set exactly one of active_ratio or active_token_budget')
    require(ratio is None or (type(ratio) is float and 0 < ratio < 1), 'active_ratio must be in (0, 1)')
    require(budget is None or (type(budget) is int and budget >= page),
            'active_token_budget must cover at least one Page')
    require(type(contract['recent_turns']) is int and contract['recent_turns'] >= 0
            and type(contract['recent_quota_pages']) is int and contract['recent_quota_pages'] >= 0,
            'recent safety bounds must be non-negative integers')
    require(contract['mandatory'] == 'initial_prompt_pages' and contract['vector_dtype'] == 'int8'
            and contract['hidden_state'] == 'pre_final_norm'
            and contract['projection_dimension'] == PROJECTION_DIMENSION
            and contract['projection_seed'] == PROJECTION_SEED, 'upstream vector space required')
    require(contract['goal_query_variant'] in GOAL_QUERY_VARIANTS, 'unknown Goal query variant')
    require(type(contract['entity_drift_steps']) is int and contract['entity_drift_steps'] > 0,
            'entity drift steps must be positive')
    rewriter = contract['goal_rewriter']
    if rewriter is not None:
        require(type(rewriter) is dict and set(rewriter) == REWRITER_KEYS
                and type(rewriter['model']) is str and Path(rewriter['model']).is_absolute()
                and type(rewriter['config_sha256']) is str and len(rewriter['config_sha256']) == 64
                and type(rewriter['max_new_tokens']) is int and rewriter['max_new_tokens'] > 0
                and all(type(rewriter[k]) is float and rewriter[k] > 0
                        for k in ('timeout_seconds', 'startup_timeout_seconds')),
                'exact Goal rewriter pin required')
    return contract


def uses_goal_rewriter(policy):
    """Only goal_query's Selector V2 query is the Goal.

    observation_query ranks Pages by the cosine of the phase-1 observation
    vector alone; its Goal reaches the selector only as a session check and the
    evidence, so it is built deterministically and never runs the rewriter.
    """
    return policy == 'goal_query'


def for_policy(contract, policy):
    """The exact contract one server policy may run: only goal_query pins a rewriter,
    and random_fill alone (always) pins its random fill."""
    validate(contract)
    require(contract['goal_rewriter'] is None or uses_goal_rewriter(policy),
            '%s does not use the Goal rewriter; its selection contract must not pin one' % policy)
    require((RANDOM_FILL in contract) == (policy == RANDOM_FILL),
            'random_fill requires the random fill pin of its selection contract' if policy == RANDOM_FILL
            else '%s does not fill at random; its selection contract must not pin a random fill' % policy)
    return contract


def rewriter_pin(model, *, max_new_tokens=384, timeout_seconds=120.0,
                 startup_timeout_seconds=300.0):
    """Pin the local Goal rewriter checkpoint by its config bytes."""
    root = Path(model).resolve(strict=True)
    config = root / 'config.json'
    require(root.is_dir() and config.is_file(), 'Goal rewriter checkpoint directory with config.json required')
    return dict(model=str(root), config_sha256=sha256(config.read_bytes()).hexdigest(),
                max_new_tokens=max_new_tokens, timeout_seconds=float(timeout_seconds),
                startup_timeout_seconds=float(startup_timeout_seconds))


def digest(contract):
    return sha256(json.dumps(validate(contract), sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def history_tokens(n, contract):
    return n * contract['page_tokens']


def selection_active(n, turn, contract):
    """Upstream keeps every Page active on the first turn and up to the threshold."""
    return turn > 0 and history_tokens(n, contract) > contract['switch_threshold_tokens']


def capacity_pages(n, mandatory_pages, contract):
    """Upstream ``_plan`` capacity in tokens, floored to whole native Pages."""
    page = contract['page_tokens']
    history = history_tokens(n, contract)
    if contract['active_token_budget'] is None:
        budget = math.floor(history * contract['active_ratio'])
    else:
        budget = contract['active_token_budget']
    return min(n, max(1, mandatory_pages, budget // page))


def mandatory_pages(initial_prompt_end, n, contract):
    """Every Page touched by the initial system + task prompt (cover, not floor)."""
    require(type(initial_prompt_end) is int and initial_prompt_end > 0, 'initial prompt boundary required')
    return min(n, -(-initial_prompt_end // contract['page_tokens']))


def page_turn(page, turn_ends, page_tokens):
    """Upstream 1-based turn of a Page = turn of its last token.

    ``turn_ends[k-1]`` is the end of upstream turn k (the stable boundary seen
    at vLLM turn k).  Tokens before the first boundary belong to turn 1.
    """
    last = (page + 1) * page_tokens - 1
    return 1 + bisect_right(turn_ends, last)


_MASK64 = (1 << 64) - 1
_GOLDEN64 = 0x9E3779B97F4A7C15


def fill_stream_key(seed, session_index, turn):
    """The 64-bit SplitMix64 seed of one turn's draw: seed * 2^32 + session * 2^16 + turn.

    ``turn`` is the session's request ordinal (it keeps counting across a
    rebuild; the native session refuses ordinals of 256 and above). Distinct
    (seed, session, turn) give distinct keys; nothing is hashed."""
    require(type(seed) is int and 0 <= seed < FILL_SEED_LIMIT and type(session_index) is int
            and 0 <= session_index < 1 << 16 and type(turn) is int and 0 <= turn < 1 << 16,
            'random fill stream key out of range')
    return seed << 32 | session_index << 16 | turn


def splitmix64(state):
    """One SplitMix64 step (Steele, Lea and Flood 2014): (next state, 64-bit output)."""
    state = (state + _GOLDEN64) & _MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, z ^ (z >> 31)


def fill_order(eligible, seed, session_index, turn):
    """A uniformly random permutation of ``eligible`` (``RANDOM_FILL_RNG``).

    SplitMix64 seeded with ``fill_stream_key``; Fisher-Yates from the front: for
    i = 0 .. m-2, swap item i with item i + (z mod (m - i)), where a 64-bit output
    z at or above the largest multiple of m - i below 2^64 is drawn again (no
    modulo bias). Its first k items are a uniform draw of k items without
    replacement. Deterministic, integer arithmetic only."""
    order = list(eligible)
    state = fill_stream_key(seed, session_index, turn)
    for i in range(len(order) - 1):
        bound = len(order) - i
        limit = (1 << 64) - (1 << 64) % bound
        state, z = splitmix64(state)
        while z >= limit:
            state, z = splitmix64(state)
        j = i + z % bound
        order[i], order[j] = order[j], order[i]
    return order


def projection(hidden_size, contract):
    import torch
    generator = torch.Generator(device='cpu')
    generator.manual_seed(contract['projection_seed'])
    values = torch.randint(0, 2, (contract['projection_dimension'], hidden_size),
                           generator=generator, dtype=torch.int8).to(torch.float32)
    values.mul_(2.0).sub_(1.0).mul_(1.0 / math.sqrt(contract['projection_dimension']))
    value = sha256(values.contiguous().numpy().tobytes()).hexdigest()
    expected = PROJECTION_SHA256.get(hidden_size)
    require(expected is None or value == expected, 'projection differs from upstream vector space')
    return values, value


def compact(values, matrix, compact_vector):
    """Upstream ``_compact_last``: project, L2 normalise, int8 per-vector scale."""
    import torch
    vector = torch.nn.functional.linear(torch.tensor(values, dtype=torch.float32), matrix)
    vector = torch.nn.functional.normalize(vector, p=2, dim=-1)
    peak = vector.abs().max()
    require(bool(torch.isfinite(peak)) and float(peak) != 0.0, 'projection produced an invalid zero vector')
    scale = peak / 127.0
    quantized = torch.clamp(torch.round(vector / scale), -127, 127).to(torch.int8)
    return compact_vector(tuple(int(v) for v in quantized.tolist()), 'int8', float(scale))


def compact_batch(rows, matrix):
    """``compact`` for a batch of rows in one pass, on the rows' device (WP3, decoupled).

    rows: [n, hidden] (any float dtype, pre-final-norm), matrix: the projection.
    Returns (int8 [n, 128], float32 scale [n]) on that device; one D2H of both
    gives the wire form.  Same math as the upstream ``_compact_last`` per row
    (fp32 linear, L2 normalise, peak / 127, round, clamp); GEMM accumulation
    order may move an element across a rounding boundary against the one-row
    path (open question 3: equal cosine ranking, not bit identity).
    """
    import torch
    require(rows.ndim == 2 and rows.shape[0] > 0, 'non-empty [n, hidden] rows required')
    vector = torch.nn.functional.linear(rows.float(), matrix.to(device=rows.device, dtype=torch.float32))
    vector = torch.nn.functional.normalize(vector, p=2, dim=-1)
    peak = vector.abs().amax(dim=-1)
    require(bool(torch.isfinite(peak).all()) and bool((peak > 0).all()),
            'projection produced an invalid zero vector')
    scale = peak / 127.0
    quantized = torch.clamp(torch.round(vector / scale[:, None]), -127, 127).to(torch.int8)
    return quantized, scale


def compact_wire(quantized, scales):
    """Host copies of ``compact_batch`` -> [[q_0..q_127], scale] per row (``compact_int8_v1``)."""
    rows, values = quantized.tolist(), scales.tolist()
    require(len(rows) == len(values), 'one scale per compact row required')
    return [[row, float(scale)] for row, scale in zip(rows, values)]


def parse_compact(value, compact_vector, dimension=PROJECTION_DIMENSION):
    """One ``compact_int8_v1`` Page vector -> ``compact_vector(values, 'int8', scale)``.

    Wire form ``[[q_0, ..., q_{d-1}], scale]``: exactly ``dimension`` ints in
    [-127, 127], not all zero, and a finite positive float scale.  An instance
    of ``compact_vector`` is accepted as is (same checks)."""
    message = 'compact_int8_v1 vector [[int8 x %d], scale] required' % dimension
    if isinstance(value, compact_vector):
        require(value.dtype == 'int8' and len(value.values) == dimension, message)
        return value
    require(type(value) in (list, tuple) and len(value) == 2 and type(value[0]) in (list, tuple)
            and len(value[0]) == dimension and type(value[1]) is float, message)
    try:
        # The upstream class checks every value (int, not bool, in [-127, 127],
        # not all zero) and the scale (finite, positive).
        return compact_vector(tuple(value[0]), 'int8', value[1])
    except (TypeError, ValueError) as exc:
        failure = str(exc)
    require(False, message + ': ' + failure)


def goal_query_text(goal, variant):
    """Upstream ``_goal_query_text``; the variant never mutates the maintained Goal."""
    retrieval = asdict(goal.retrieval_goal)
    if variant == 'candidate_entity_cap8':
        for field in ('files', 'symbols', 'tests'):
            retrieval[field] = retrieval[field][-8:]
    else:
        require(variant == 'full_goal', 'unknown Goal query variant')
    return '<retrieval_query>\n' + json.dumps({'retrieval_goal': retrieval}, ensure_ascii=False,
        sort_keys=True, separators=(',', ':')) + '\n</retrieval_query>'


def constructor_control_hash(constructor):
    """Upstream ``goal_control_hash`` fed to the selector as frozen trace hash."""
    raw = json.dumps({'constructor': constructor.name, 'config': constructor._config()},
                     ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return sha256(raw.encode('utf-8')).hexdigest()
