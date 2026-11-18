"""TRITON_ATTN on the Qwen profile: ``serve.py --triton-attn``, off by default, checked once at startup.

Without the switch vLLM 0.24.0 picks FlashAttention 3 for Qwen3.6-27B's 16 full-attention
layers on a GH200, as before. With it the vLLM command gains ``--attention-backend
TRITON_ATTN`` (vLLM's Triton ``unified_attention``; the Gated-DeltaNet layers keep their own
backend and the Triton kernels of ``--gdn-prefill-backend triton``), so a server can match
baselines that run TRITON_ATTN. Nothing in the runtime depends on the backend's kernels:

- the runner rewrites vLLM's ``CommonAttentionMetadata`` (the selective physical seq lens,
  the compact block tables and slot mapping) before any backend's metadata builder runs,
  and both builders read only those fields (TRITON_ATTN has no AOT scheduler metadata and
  never uses cascade attention);
- the KV views come from the loaded backend's ``get_kv_cache_shape``
  (``remote_ra_growing_worker.tensors.views_from_runner``); both backends lay a layer's
  cache out as ``(num_blocks, 2, block_size, num_kv_heads, head_size)`` with the same stride
  orders: NHD, and HND, which vLLM uses here because the runtime's OffloadingConnector asks for
  it (the Gemma 4 profile already runs TRITON_ATTN with it); both kernels index the cache by
  its strides;
- both write the KV through the same splitting op (``vllm::unified_kv_cache_update``) and
  attend through ``vllm::unified_attention_with_output``, so the CUDA-graph modes, their
  splitting ops and the FULL graphs' physical seq-len binding are the same; TRITON_ATTN
  supports every graph (``AttentionCGSupport.ALWAYS``), its split-KV decode buffers are
  allocated once by its builder;
- the decoupled layout's Pages are kernel blocks of both backends (FlashAttention on this
  hybrid fp32-state model takes 16, 32 or 64; TRITON_ATTN any multiple of 16), so the
  tensor geometry is the same. The shared pool's 784-token blocks are not: FlashAttention
  splits them into 49 kernel blocks of 16, TRITON_ATTN would not; the switch is refused
  there (and at the decoupled layout's validation size 784).

The Gemma 4 profile always runs TRITON_ATTN (model_profile.GEMMA4_ATTENTION_BACKEND, pinned
by its own startup check); there the switch changes nothing and this key is never written.

Off (the default): no vLLM argument, startup config key, worker startup record key or
evidence key; the frontend, the engine and the worker refuse a Qwen vLLM that resolved
TRITON_ATTN anyway. On: ``startup.json`` pins ``attention_backend: "TRITON_ATTN"`` and
``effective_server_config.attention_backend``, which the frontend compares with vLLM's
parsed argument; ``compose_engine`` checks the engine's resolved attention config and sends
the pin once in the worker startup record; the worker checks its runner's resolved config
and that every FullAttention group loaded vLLM's Triton backend, and echoes ``RECORD`` in its
attach evidence, which the engine and ``write_ready`` compare; ``ready.json`` records it
(next to ``source_closure.backends``, which names and hashes the loaded backend module on
either backend). Nothing here runs per request or step.
"""
from __future__ import annotations

import os

from .contract import require

CONFIG_KEY = 'attention_backend'    # startup config, effective_server_config, worker record, evidence, ready.json
ARGUMENT = '--attention-backend'    # vLLM's flag; serve.py passes it with TRITON
SWITCH = '--triton-attn'            # serve.py's opt-in
TRITON = 'TRITON_ATTN'              # vLLM's AttentionBackendEnum member
MODULE = 'vllm.v1.attention.backends.triton_attn'   # TritonAttentionBackend's module
PAGE_TOKENS = (16, 32, 64)          # decoupled Pages that are kernel blocks of both backends
# The worker's attach evidence with the switch on: what the worker checked.
RECORD = {'backend': TRITON, 'module': MODULE}


def _name(value):
    """An enum member by name (vLLM's AttentionBackendEnum); text as it is; None stays None."""
    name = getattr(value, 'name', None)
    return name if isinstance(name, str) else value


def pinned(config):
    """'TRITON_ATTN' when a Qwen startup config pins --triton-attn, else None (the key is absent).

    The pin comes with exactly its vLLM argument in effective_server_config and the
    decoupled layout at 16-, 32- or 64-token Pages. Without the key a Qwen config pins no
    attention backend; a Gemma 4 config (``model_profile``) pins its own and never this key."""
    if config is None:
        return None
    effective = config.get('effective_server_config')
    gemma4 = 'model_profile' in config
    if CONFIG_KEY not in config:
        require(gemma4 or not isinstance(effective, dict) or CONFIG_KEY not in effective,
                'attention_backend needs --triton-attn: the startup config pins vLLM\'s own choice')
        return None
    require(not gemma4, 'the Gemma 4 profile pins TRITON_ATTN through its profile, not attention_backend')
    require(config[CONFIG_KEY] == TRITON,
            'pinned attention_backend must be TRITON_ATTN; vLLM\'s own choice is pinned by leaving it out')
    require(isinstance(effective, dict) and effective.get(CONFIG_KEY) == TRITON,
            'attention_backend differs from the pinned vLLM arguments')
    require(config.get('page_tokens') in PAGE_TOKENS and config.get('kv_layout') is not None,
            'attention_backend TRITON_ATTN needs the decoupled KV layout at 16-, 32- or 64-token Pages')
    return TRITON


def record(config):
    """The worker evidence record a startup config expects (RECORD), or None (off)."""
    return dict(RECORD) if pinned(config) else None


def loaded():
    """The pin of this process's startup config on the HTTP runtime, else None. Startup only."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def resolved(vllm_config):
    """vLLM's resolved --attention-backend by name (None: vLLM chooses, or no config to read)."""
    return _name(getattr(getattr(vllm_config, 'attention_config', None), 'backend', None))


def parsed(value):
    """The frontend: vLLM's parsed --attention-backend by name (its enum, text or None)."""
    return _name(value)


def check_engine(vllm_config, config):
    """compose_engine (HTTP runtime): the engine's resolved backend agrees with the pin.

    A Gemma 4 config is left to its profile's checks (it always resolves TRITON_ATTN)."""
    if config is not None and 'model_profile' in config:
        return None
    pin = pinned(config)
    backend = resolved(vllm_config)
    if pin is None:
        require(backend != TRITON, 'vLLM runs TRITON_ATTN but the startup config pins vLLM\'s own choice')
        return None
    require(backend == TRITON, 'vLLM resolved attention backend %r, pinned TRITON_ATTN' % (backend,))
    return TRITON


def full_attention_backends(runner):
    """(group id, backend name, backend module) of every FullAttention group the runner built."""
    groups = getattr(runner, '_kv_cache_spec_attn_group_iterator', None)
    require(callable(groups), 'TRITON_ATTN check needs the loaded runner\'s attention groups')
    found = []
    for group in groups():
        if type(group.kv_cache_spec).__name__ != 'FullAttentionSpec':
            continue
        backend = group.backend
        found.append((int(group.kv_cache_group_id), backend.get_name(), backend.__module__))
    return found


def worker_check(runner, startup):
    """Worker startup: the record the engine sent, checked once against the pinned config
    and against the loaded runner. Returns RECORD (on) or None (off).

    With the HTTP runtime the worker reads the startup config itself and refuses a
    record that differs; without it (CPU fixtures) only the record's value is checked."""
    sent = startup.get(CONFIG_KEY)
    require(sent is None or sent == TRITON, 'attention_backend worker record must be TRITON_ATTN or absent')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker attention_backend differs from the pinned startup config')
    backend = resolved(getattr(runner, 'vllm_config', None))
    if sent is None:
        require(backend != TRITON, 'vLLM runs TRITON_ATTN but the startup config pins vLLM\'s own choice')
        return None
    require(backend == TRITON, 'the worker\'s vLLM resolved attention backend %r, pinned TRITON_ATTN' % (backend,))
    groups = full_attention_backends(runner)
    require(groups, 'TRITON_ATTN check found no FullAttention group')
    wrong = [g for g in groups if g[1:] != (TRITON, MODULE)]
    require(not wrong, 'FullAttention groups loaded another attention backend: %r' % (wrong,))
    return dict(RECORD)
