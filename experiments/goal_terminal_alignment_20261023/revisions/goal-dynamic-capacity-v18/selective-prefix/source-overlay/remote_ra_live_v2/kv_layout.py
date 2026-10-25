"""KV layout shared by vLLM and the runtime (interface I0, exp-decouple-pages).

Pure Python, no imports from the runtime or vLLM, so the vLLM overlay can
import it lazily from any process.

- Legacy (default): one shared block pool, FA block == Mamba block == 784
  tokens, the Mamba page padded to the FA page. No additional_config key.
- Decoupled: two block pools. Pool 0 holds FA blocks of P tokens (one block
  per runtime Page); pool 1 holds unpadded GDN states, one block per Mamba
  group per slot, K slots. M is the Mamba align block (the GDN crossing
  cadence), a multiple of P.

The switch is ``VllmConfig.additional_config[LAYOUT_KEY]`` =
``{"page_tokens": P, "mamba_tokens": M, "mamba_slots": K}``;
``to_additional_config`` returns the whole additional_config dict
(``{LAYOUT_KEY: {...}}``), or None for legacy, and ``from_vllm_config``
inverts it. Everything here is evaluated at startup only.

- Window (gemma-s1, the Gemma 4 model profile only): ``WindowLayout``. Pool 0
  holds the global full-attention layers in blocks of P tokens (one block per
  runtime Page); pool 1 holds the sliding-window layers in blocks of P tokens,
  statically reserved: a null block plus, per sliding group, S window slots of
  ceil(W / P) + 1 blocks (W = the window) and one forward's transient
  ceil((W - 1 + N) / P) + 1 - (ceil(W / P) + 1) blocks (N = the largest native
  forward; vLLM's own per-request bound). Its switch carries
  ``{"page_tokens": P, "state": "sliding", "window_tokens": W,
  "window_slots": S, "forward_tokens": N}``; the Mamba form above is unchanged.
  A goal_query server (gemma-goal) also reserves the Goal forward's own transient:
  per sliding group ceil(G / P) blocks for a cold Goal forward of at most G tokens,
  which runs after phase 1 while phase 1's fresh blocks are still held; its switch
  adds ``"goal_tokens": G`` (absent, and nothing reserved, on every other server).
  A server whose launcher sized pool 0 by the selective rule (gemma-mab:
  --release-idle-pages with a query policy and a fixed --active-token-budget) adds
  ``"request_pages": R``: vLLM's own startup check then asks pool 0 for R global Pages
  for one request, the selective rule's figure, instead of one request at
  max_model_len (the runtime's capacity gate never lets a turn take more: a first turn
  that cannot fit is refused, idle_release's CAPACITY_EXCEEDED). Absent on every
  other server, where the check is vLLM's.
"""
from __future__ import annotations

from dataclasses import dataclass

LAYOUT_KEY = 'remote_ra_kv_layout'   # key in VllmConfig.additional_config
LEGACY_PAGE = 784
MAX_MAMBA_TOKENS = 16384             # M <= max_num_batched_tokens of the pinned profile
_FIELDS = ('page_tokens', 'mamba_tokens', 'mamba_slots')


def _integer(name, value):
    if type(value) is not int:
        raise ValueError(f'{LAYOUT_KEY}.{name} must be an int, got {type(value).__name__}')
    return value


@dataclass(frozen=True)
class KvLayout:
    page_tokens: int     # P: FA block size == runtime Page, multiple of 16
    mamba_tokens: int    # M: Mamba align block (GDN crossing cadence), multiple of P and 16
    mamba_slots: int     # K: GDN slots in the Mamba pool (1 slot = 1 block per Mamba group); 0 in legacy
    decoupled: bool      # False only for the legacy shared pool (P == M == 784, K == 0)

    def __post_init__(self):
        P = _integer('page_tokens', self.page_tokens)
        M = _integer('mamba_tokens', self.mamba_tokens)
        K = _integer('mamba_slots', self.mamba_slots)
        if type(self.decoupled) is not bool:
            raise ValueError('KvLayout.decoupled must be a bool')
        if P <= 0 or P % 16:
            raise ValueError(f'page_tokens must be a positive multiple of 16, got {P}')
        if not 0 < M <= MAX_MAMBA_TOKENS or M % 16 or M % P:
            raise ValueError(f'mamba_tokens must be a multiple of 16 and of page_tokens '
                             f'in (0, {MAX_MAMBA_TOKENS}], got {M} (page_tokens {P})')
        if self.decoupled:
            if K < 3:
                raise ValueError(f'decoupled layout needs mamba_slots >= 3, got {K}')
        elif (P, M, K) != (LEGACY_PAGE, LEGACY_PAGE, 0):
            raise ValueError(f'legacy layout requires page_tokens == mamba_tokens == '
                             f'{LEGACY_PAGE} and mamba_slots == 0, got {(P, M, K)}')


SLIDING = 'sliding'
_WINDOW_FIELDS = ('page_tokens', 'state', 'window_tokens', 'window_slots', 'forward_tokens')
GOAL_FIELD = 'goal_tokens'           # the Goal transient (goal_query only); absent means 0
REQUEST_FIELD = 'request_pages'      # the selective rule's Pages for vLLM's startup check; absent means 0
MAX_WINDOW_SLOTS = 64


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class WindowLayout:
    """Two pools for a sliding-window model (gemma-s1): global Pages and sliding windows."""
    page_tokens: int     # P: the block size of every KV group (vLLM --block-size), one runtime Page
    window_tokens: int   # W: the sliding window of the sliding layers
    window_slots: int    # S: window slots in pool 1 (one per open session, the window kept on the GPU)
    forward_tokens: int  # N: the largest native forward (max_num_batched_tokens)
    goal_tokens: int = 0  # G: the largest Goal forward of a goal_query server (its own transient); 0: none
    request_pages: int = 0  # R: pool-0 Pages vLLM's startup check asks for one request; 0: one at max_model_len

    decoupled = True
    state = SLIDING

    def __post_init__(self):
        P = _integer('page_tokens', self.page_tokens)
        W = _integer('window_tokens', self.window_tokens)
        S = _integer('window_slots', self.window_slots)
        N = _integer('forward_tokens', self.forward_tokens)
        G = _integer(GOAL_FIELD, self.goal_tokens)
        R = _integer(REQUEST_FIELD, self.request_pages)
        if P <= 0 or P % 16:
            raise ValueError(f'page_tokens must be a positive multiple of 16, got {P}')
        if W <= 0 or W % P:
            raise ValueError(f'window_tokens must be a positive multiple of page_tokens {P}, got {W}')
        if not 1 <= S <= MAX_WINDOW_SLOTS:
            raise ValueError(f'window_slots must be in [1, {MAX_WINDOW_SLOTS}], got {S}')
        if N < P or N % P or N > MAX_MAMBA_TOKENS:
            raise ValueError(f'forward_tokens must be a multiple of page_tokens {P} in '
                             f'[{P}, {MAX_MAMBA_TOKENS}], got {N}')
        if G and (G < P or G % P or G > N):
            raise ValueError(f'goal_tokens must be 0 or a multiple of page_tokens {P} in [{P}, {N}] '
                             f'(one native forward), got {G}')
        if R < 0:
            raise ValueError(f'request_pages must be 0 or positive, got {R}')

    @property
    def window_blocks(self) -> int:
        """Blocks of one window slot per sliding group: a query at position n reads
        [n - W + 1, n], which spans at most ceil(W / P) + 1 blocks."""
        # One extra physical block protects the accepted EOS window while native
        # async decoding can be at most two tokens ahead. Attention W is unchanged.
        return _cdiv(self.window_tokens, self.page_tokens) + 2

    @property
    def transient_blocks(self) -> int:
        """Blocks per sliding group that one forward of N new tokens holds beyond a
        window slot (vLLM: ceil((W - 1 + N) / P) + 1 blocks per request)."""
        return _cdiv(self.window_tokens - 1 + self.forward_tokens, self.page_tokens) + 1 - self.window_blocks

    @property
    def goal_blocks(self) -> int:
        """Blocks per sliding group of the Goal forward's own transient: a cold forward of
        at most G tokens holds every one of its blocks, ceil(G / P) (0 without a Goal)."""
        return _cdiv(self.goal_tokens, self.page_tokens)


def is_window(layout) -> bool:
    """Whether a decoupled layout is the sliding-window form (gemma-s1)."""
    return isinstance(layout, WindowLayout)


def for_window_server(P: int, sessions: int, window_tokens: int, forward_tokens: int,
                      goal_tokens: int = 0) -> WindowLayout:
    """Server layout of a sliding-window model: one window slot per session (W-GPU) and,
    for goal_query (``goal_tokens`` G > 0), the Goal forward's transient."""
    if type(sessions) is not int or sessions < 1:
        raise ValueError(f'sessions must be a positive int, got {sessions!r}')
    return WindowLayout(P, window_tokens, sessions, forward_tokens, goal_tokens)


def window_pool_blocks(layout: WindowLayout, sliding_groups: int) -> int:
    """Blocks of the window pool: null block + per sliding group S slots, one forward's
    transient and the Goal forward's transient (none unless goal_tokens > 0)."""
    if not is_window(layout):
        raise ValueError('only the window layout has a window pool')
    if type(sliding_groups) is not int or sliding_groups < 1:
        raise ValueError(f'sliding_groups must be a positive int, got {sliding_groups!r}')
    return 1 + sliding_groups * (layout.window_slots * layout.window_blocks + layout.transient_blocks
                                 + layout.goal_blocks)


def legacy() -> KvLayout:
    return KvLayout(LEGACY_PAGE, LEGACY_PAGE, 0, False)


def default_mamba_tokens(P: int) -> int:
    """M = P * ceil(784 / P): 784, 800, 832, 784 for P = 16, 32, 64, 784."""
    _integer('page_tokens', P)
    if P <= 0 or P % 16:
        raise ValueError(f'page_tokens must be a positive multiple of 16, got {P}')
    return P * -(-LEGACY_PAGE // P)


def for_server(P: int, sessions: int, decoupled: bool | None = None) -> KvLayout:
    """Server layout: decoupled defaults to P != 784; K = 2 * sessions + 1."""
    _integer('page_tokens', P)
    if type(sessions) is not int or sessions < 1:
        raise ValueError(f'sessions must be a positive int, got {sessions!r}')
    if decoupled is None:
        decoupled = P != LEGACY_PAGE
    if not decoupled:
        if P != LEGACY_PAGE:
            raise ValueError(f'page_tokens {P} requires the decoupled layout')
        return legacy()
    return KvLayout(P, default_mamba_tokens(P), 2 * sessions + 1, True)


def to_additional_config(layout: KvLayout) -> dict | None:
    """The whole --additional-config dict; None for legacy (serve.py emits no flag)."""
    if is_window(layout):
        record = {'page_tokens': layout.page_tokens, 'state': SLIDING,
                  'window_tokens': layout.window_tokens,
                  'window_slots': layout.window_slots,
                  'forward_tokens': layout.forward_tokens}
        if layout.goal_tokens:
            record[GOAL_FIELD] = layout.goal_tokens
        if layout.request_pages:
            record[REQUEST_FIELD] = layout.request_pages
        return {LAYOUT_KEY: record}
    if not layout.decoupled:
        return None
    return {LAYOUT_KEY: {'page_tokens': layout.page_tokens,
                         'mamba_tokens': layout.mamba_tokens,
                         'mamba_slots': layout.mamba_slots}}


def from_vllm_config(vllm_config) -> KvLayout:
    """Legacy if the key is absent; otherwise the validated decoupled layout
    (a WindowLayout when the record names the sliding state)."""
    extra = getattr(vllm_config, 'additional_config', None)
    if not isinstance(extra, dict) or LAYOUT_KEY not in extra:
        return legacy()
    value = extra[LAYOUT_KEY]
    if isinstance(value, dict) and 'state' in value:
        if set(value) - {GOAL_FIELD, REQUEST_FIELD} != set(_WINDOW_FIELDS) or value['state'] != SLIDING:
            raise ValueError(f'{LAYOUT_KEY} with a state must be the sliding window form with exactly '
                             f'the keys {_WINDOW_FIELDS} (and {GOAL_FIELD!r} for a Goal transient, '
                             f'{REQUEST_FIELD!r} for the selective rule), got {value!r}')
        goal = _integer(GOAL_FIELD, value.get(GOAL_FIELD, 0))
        if GOAL_FIELD in value and goal <= 0:
            raise ValueError(f'{LAYOUT_KEY}.{GOAL_FIELD} must be positive; no Goal transient is '
                             f'pinned by leaving it out, got {value!r}')
        request = _integer(REQUEST_FIELD, value.get(REQUEST_FIELD, 0))
        if REQUEST_FIELD in value and request <= 0:
            raise ValueError(f'{LAYOUT_KEY}.{REQUEST_FIELD} must be positive; vLLM\'s own check is '
                             f'pinned by leaving it out, got {value!r}')
        return WindowLayout(*(_integer(name, value[name]) for name in _WINDOW_FIELDS if name != 'state'), goal,
                            request)
    if not isinstance(value, dict) or set(value) != set(_FIELDS):
        raise ValueError(f'{LAYOUT_KEY} must be a dict with exactly the keys {_FIELDS}, '
                         f'got {value!r}')
    return KvLayout(*(_integer(name, value[name]) for name in _FIELDS), True)


def mamba_pool_blocks(layout: KvLayout, mamba_groups: int) -> int:
    """Blocks of the decoupled Mamba pool: null block + one block per group per slot."""
    if is_window(layout):
        raise ValueError('the window layout has no Mamba pool')
    if not layout.decoupled:
        raise ValueError('the legacy layout has no separate Mamba pool')
    if type(mamba_groups) is not int or mamba_groups < 1:
        raise ValueError(f'mamba_groups must be a positive int, got {mamba_groups!r}')
    return 1 + mamba_groups * layout.mamba_slots


def mamba_row_len(end: int, layout: KvLayout) -> int:
    """Mamba block-table entries covering tokens [0, end): ceil(end / M)."""
    if type(end) is not int or end < 0:
        raise ValueError(f'end must be a non-negative int, got {end!r}')
    return -(-end // layout.mamba_tokens)


def mamba_index(end: int, layout: KvLayout) -> int:
    """Align-mode slot index of the running GDN state after `end` tokens: ceil(end / M) - 1."""
    if type(end) is not int or end < 1:
        raise ValueError(f'end must be a positive int, got {end!r}')
    return mamba_row_len(end, layout) - 1
