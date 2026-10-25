"""CUDA graphs for the runtime: one startup flag, verified once at startup.

``serve.py --cuda-graphs {off,piecewise,full,full-decode}``. ``off`` (the
default) is the eager server of b37044f, byte for byte: ``--enforce-eager``,
no startup key, no worker record, no evidence key. The others drop
``--enforce-eager`` (async scheduling stays off) and pin vLLM's
``--compilation-config``:

- ``piecewise``: ``{"mode": "VLLM_COMPILE", "cudagraph_mode": "PIECEWISE"}``.
  vLLM 0.24 compiles the model and captures the pieces between its splitting
  ops. FlashAttention (``unified_attention_with_output``), the KV-cache update
  (``unified_kv_cache_update``) and the GDN core (``qwen_gdn_attention_core``)
  are splitting ops: they run eagerly every step and read this step's
  metadata (selective physical seq lens, compact slot mapping, rewritten
  block tables, GDN state indices). The runtime needs no static buffer.
- ``full``: ``{"mode": "VLLM_COMPILE", "cudagraph_mode": "FULL_AND_PIECEWISE"}``
  (stock vLLM's default). Uniform decode steps replay one graph that also
  holds attention; mixed steps stay piecewise. A captured graph keeps the
  ``seq_lens`` pointer it saw at capture, so the patched runner binds every
  FullAttention group to the physical seq-len buffer at capture and replay
  alike (``gpu_model_runner._selective_cudagraph_groups``).
- ``full-decode``: ``{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}``.
  No torch.compile: uniform decode steps replay one graph of the eager
  kernels (vLLM's custom ops, as with --enforce-eager); prefill, mixed steps
  and phase 1 run eagerly as today. The same seq-len binding as ``full``.

In every mode a probe-only pass (phase 1 and the Goal forward) runs without a
CUDA graph (``CUDAGraphMode.NONE``, unpadded), and every runtime hook stays
outside ``_model_forward``. Everything here runs at startup; the per-step
hook ``force_eager`` is one attribute check.

The window layout (Gemma 4, ``window_worker``; TRITON_ATTN): its splitting ops
are attention and the KV-cache update (``WINDOW_SPLITTING_OPS``; there is no
GDN layer). In ``full`` and ``full-decode`` the runner binds its global group,
the only FullAttention group (vLLM's SlidingWindowSpec is not a
FullAttentionSpec), to the physical seq-len buffer, and that group keeps the
physical slot mapping; the five sliding groups keep vLLM's logical seq_lens,
positions and slot mapping. Every buffer a captured graph reads is one of the
runner's persistent buffers (seq_lens, the physical seq_lens, positions,
query_start_loc, each group's block table and slot mapping), rewritten in
place before every replay.
"""
from __future__ import annotations

import json

from .contract import require

CONFIG_KEY = 'cuda_graphs'                 # startup config, worker startup record, worker evidence
MODES = ('off', 'piecewise', 'full', 'full-decode')   # also written out in serve.py's --cuda-graphs choices
CUDAGRAPH_MODE = {'piecewise': 'PIECEWISE', 'full': 'FULL_AND_PIECEWISE', 'full-decode': 'FULL_DECODE_ONLY'}
COMPILATION_MODE = {'piecewise': 'VLLM_COMPILE', 'full': 'VLLM_COMPILE', 'full-decode': 'NONE'}
# Modes whose captured graphs hold FlashAttention (the runner binds the seq lens).
FULL_MODES = ('full', 'full-decode')
# vLLM 0.24 ops that must stay outside every captured piece (they read the
# runtime's per-step metadata); vLLM's default splitting ops contain all three.
# Only compiled modes split; full-decode captures whole decode steps.
REQUIRED_SPLITTING_OPS = ('vllm::unified_attention_with_output',
                          'vllm::unified_kv_cache_update',
                          'vllm::qwen_gdn_attention_core')
# The window layout (Gemma 4, gemma-perf) has no GDN layer: attention (TRITON_ATTN, global
# and sliding layers alike) and the KV-cache update are the ops it needs outside the pieces.
WINDOW_SPLITTING_OPS = REQUIRED_SPLITTING_OPS[:2]
PROBE_FORWARD = 'eager'                    # probe-only passes: CUDAGraphMode.NONE


def compilation_config(mode):
    """The pinned ``--compilation-config`` of a CUDA-graph server (a dict)."""
    require(mode in CUDAGRAPH_MODE, 'CUDA-graph mode must be piecewise, full or full-decode, got %r' % (mode,))
    return {'cudagraph_mode': CUDAGRAPH_MODE[mode], 'mode': COMPILATION_MODE[mode]}


def argument(mode):
    """The ``--compilation-config`` argv value: compact JSON, sorted keys."""
    return json.dumps(compilation_config(mode), sort_keys=True, separators=(',', ':'))


def _name(value):
    """An enum member by name (vLLM's CompilationMode / CUDAGraphMode); text upper-cased."""
    if value is None:
        return None
    name = getattr(value, 'name', None)
    if isinstance(name, str):
        return name
    return value.upper() if isinstance(value, str) else value


def observed(value):
    """The pinned keys of vLLM's parsed ``--compilation-config``.

    vLLM's parser turns the JSON into a CompilationConfig; a dict or the JSON
    text are accepted too. Enum members compare by name."""
    if isinstance(value, (str, bytes)):
        value = json.loads(value)
    if isinstance(value, dict):
        get = value.get
    else:
        def get(key):
            return getattr(value, key, None)
    return {key: _name(get(key)) for key in ('cudagraph_mode', 'mode')}


def pinned(config):
    """The startup config's CUDA-graph mode ('piecewise', 'full', 'full-decode'), or None (off).

    Startup only. A mode must come with exactly its vLLM arguments in
    ``effective_server_config``: ``enforce_eager`` False and the pinned
    ``compilation_config``. Without the key, no compilation config may be
    pinned and eager stays pinned (when the effective config names it)."""
    effective = config.get('effective_server_config')
    if CONFIG_KEY not in config:
        require(not isinstance(effective, dict) or
                ('compilation_config' not in effective and effective.get('enforce_eager', True) is True),
                'CUDA graphs require --cuda-graphs: the startup config pins eager')
        return None
    mode = config[CONFIG_KEY]
    require(mode in CUDAGRAPH_MODE, 'pinned CUDA-graph mode must be piecewise, full or full-decode, got %r'
            % (mode,))
    require(isinstance(effective, dict) and effective.get('enforce_eager') is False and
            effective.get('compilation_config') == compilation_config(mode),
            'CUDA-graph mode differs from the pinned vLLM arguments')
    return mode


def engine_mode():
    """The pinned mode for the worker startup record (compose_engine), or None (off)."""
    import os
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def resolved(compilation, required=REQUIRED_SPLITTING_OPS):
    """What vLLM resolved: the loaded runner's CompilationConfig as plain JSON values.

    ``required``: the splitting ops the layout needs (``WINDOW_SPLITTING_OPS`` for the
    window layout); the record names those vLLM split."""
    sizes = getattr(compilation, 'cudagraph_capture_sizes', None) or ()
    ops = getattr(compilation, 'splitting_ops', None) or ()
    return dict(compilation_mode=_name(getattr(compilation, 'mode', None)),
                cudagraph_mode=_name(getattr(compilation, 'cudagraph_mode', None)),
                capture_sizes=[int(s) for s in sizes],
                max_capture_size=getattr(compilation, 'max_cudagraph_capture_size', None),
                backend=getattr(compilation, 'backend', None),
                custom_ops=[str(op) for op in (getattr(compilation, 'custom_ops', None) or ())],
                splitting_ops=[op for op in required if op in ops])


def check_resolved(mode, record, required=REQUIRED_SPLITTING_OPS):
    require(record['compilation_mode'] == COMPILATION_MODE[mode] and
            record['cudagraph_mode'] == CUDAGRAPH_MODE[mode],
            'vLLM resolved compilation %s / cudagraph %s, pinned %s / %s'
            % (record['compilation_mode'], record['cudagraph_mode'], COMPILATION_MODE[mode], CUDAGRAPH_MODE[mode]))
    if COMPILATION_MODE[mode] == 'VLLM_COMPILE':
        require(record['splitting_ops'] == list(required),
                'FlashAttention, the KV-cache update and the GDN core must be splitting ops '
                '(run outside the graphs)' if required == REQUIRED_SPLITTING_OPS else
                'attention and the KV-cache update must be splitting ops (run outside the graphs)')
    sizes = record['capture_sizes']
    require(sizes and all(type(s) is int and s > 0 for s in sizes) and sizes == sorted(set(sizes)) and
            record['max_capture_size'] == sizes[-1], 'CUDA-graph capture sizes invalid')


def attach(runner, mode, attention_groups, required=REQUIRED_SPLITTING_OPS):
    """Worker startup: check the loaded runner against the pinned mode, once.

    Off (None): returns None; a runner that captured CUDA graphs anyway is
    refused. On: the resolved CompilationConfig must be the pinned one, with
    attention/KV-update/GDN as splitting ops when compiled (``required``; the
    window layout's ``WINDOW_SPLITTING_OPS`` have no GDN op). ``full`` and
    ``full-decode`` also need every FullAttention group bound to the physical
    seq-len buffer (the window layout: its global group; the sliding groups
    keep vLLM's logical buffer); ``piecewise`` binds none. The returned
    record goes into the worker evidence; the runner keeps a copy for
    ``force_eager``."""
    compilation = getattr(runner, 'compilation_config', None)
    if mode is None:
        if compilation is not None:
            require(_name(getattr(compilation, 'cudagraph_mode', None)) in (None, 'NONE'),
                    'vLLM runs CUDA graphs but the startup pins eager (--cuda-graphs off)')
        return None
    require(mode in CUDAGRAPH_MODE and compilation is not None,
            'CUDA-graph mode %r needs the loaded compilation config' % (mode,))
    record = resolved(compilation, required)
    check_resolved(mode, record, required)
    bound = sorted(int(g) for g in (getattr(runner, '_selective_cudagraph_groups', None) or ()))
    if mode in FULL_MODES:
        require(bound and set(attention_groups) <= set(bound),
                'FULL CUDA graphs need every FullAttention group bound to the physical seq-len buffer')
    else:
        require(not bound, 'piecewise CUDA graphs bind no seq-len buffer')
    require(getattr(runner, '_remote_ra_cuda_graphs', None) is None, 'CUDA-graph mode already attached')
    record = dict(mode=mode, **record, seq_lens_bound_groups=bound, probe_forward=PROBE_FORWARD)
    runner._remote_ra_cuda_graphs = dict(record)
    return record


def force_eager(runner):
    """Runner hook: True runs this step without a CUDA graph (a probe-only pass while graphs are on)."""
    if getattr(runner, '_remote_ra_cuda_graphs', None) is None:
        return False
    from .native_probe import active_probe
    return active_probe(runner) is not None
