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
    if not layout.decoupled:
        return None
    return {LAYOUT_KEY: {'page_tokens': layout.page_tokens,
                         'mamba_tokens': layout.mamba_tokens,
                         'mamba_slots': layout.mamba_slots}}


def from_vllm_config(vllm_config) -> KvLayout:
    """Legacy if the key is absent; otherwise the validated decoupled layout."""
    extra = getattr(vllm_config, 'additional_config', None)
    if not isinstance(extra, dict) or LAYOUT_KEY not in extra:
        return legacy()
    value = extra[LAYOUT_KEY]
    if not isinstance(value, dict) or set(value) != set(_FIELDS):
        raise ValueError(f'{LAYOUT_KEY} must be a dict with exactly the keys {_FIELDS}, '
                         f'got {value!r}')
    return KvLayout(*(_integer(name, value[name]) for name in _FIELDS), True)


def mamba_pool_blocks(layout: KvLayout, mamba_groups: int) -> int:
    """Blocks of the decoupled Mamba pool: null block + one block per group per slot."""
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
