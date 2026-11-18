"""Text-only serving: ``serve.py --language-model-only``, off by default, checked once at startup.

Qwen3.6-27B is a multimodal checkpoint (``Qwen3_5ForConditionalGeneration``). Without
the flag stock vLLM 0.24.0 builds and loads its vision tower, profiles the encoder
cache with a dummy image, and the API server warms the multimodal processor with
fake image and video inputs sized to max_model_len before it is ready; that warmup
briefly takes tens of GiB of host memory. This runtime serves text only.

vLLM's own ``--language-model-only`` sets every multimodal limit to 0, so vLLM runs
in text-only mode (its log: "All limits of multimodal modalities supported by the
model are set to 0, running in text-only mode"). In the pinned vLLM 0.24.0 sources:

- ``_mark_tower_model`` puts a placeholder in place of a tower whose modalities all
  have limit 0 (``vllm/model_executor/models/interfaces.py``); ``qwen3_5.py`` builds
  ``self.visual`` inside it, so no vision-tower weight is created or loaded;
- without multimodal inputs the renderer creates no multimodal processor, so the
  API server's multimodal warmups do not run (``vllm/renderers/base.py``), and
  the scheduler, input processor and runner set up no encoder cache or encoder
  profiling;
- the runner passes the model token ids instead of the embeddings it looked up
  itself (``gpu_model_runner._preprocess``): the same lookup, done in the model;
- the language model's weights are the same. Its 16 full-attention layers run
  vLLM's fused QK-RMSNorm + RoPE + gate kernel, which vLLM enables in text-only
  mode only (``qwen3_next.Qwen3NextAttention``), instead of the separate ops:
  the same arithmetic on identical mRoPE rows (``positions.text_positions``
  checks every step that the three rows are the logical positions), but not
  necessarily the same bits as a server without the flag. Pass the same flag to
  the stock vLLM control.

The KV cache is sized by ``--kv-cache-memory-bytes`` (serve.py), not by the memory
left after the weights, so its blocks and pools do not change; the GPU keeps the
vision tower's weights and the encoder cache free.

Off (the default): no vLLM argument, startup config key, worker startup record
key or evidence key; the engine and the worker refuse a vLLM that resolved
language-model-only anyway. On: the vLLM command gains ``--language-model-only``;
``startup.json`` pins ``language_model_only: true`` and
``effective_server_config.language_model_only: true``, which the frontend compares
with vLLM's parsed argument (``remote_ra_http.config.effective_server_config``);
``compose_engine`` checks the engine's resolved multimodal config and sends the
pin once in the worker startup record; the worker checks its runner's resolved
config and that no vision-tower tensor is loaded, and echoes ``RECORD`` in its
attach evidence, which the engine and ``write_ready`` compare; ``ready.json``
records it. Nothing here runs per request or step.
"""
from __future__ import annotations

import os

from .contract import require

CONFIG_KEY = 'language_model_only'   # startup config, effective_server_config, worker record, evidence, ready.json
ARGUMENT = '--language-model-only'   # vLLM's flag (serve.py appends it to the command)
TOWER = 'visual'                     # Qwen3_5ForConditionalGeneration.visual, the vision tower
TOWER_MODALITIES = ('image', 'video')  # the tower's modalities (qwen3_5.py: _mark_tower_model)
# The worker's attach evidence with the flag on: what the worker checked.
RECORD = {'mm_limits': {m: 0 for m in TOWER_MODALITIES}, 'vision_tower_tensors': 0}


def pinned(config):
    """True when a startup config pins --language-model-only, else None (the key is absent).

    The flag comes with exactly its vLLM argument in effective_server_config;
    without the key, no such argument may be pinned."""
    effective = None if config is None else config.get('effective_server_config')
    if config is None or CONFIG_KEY not in config:
        require(not isinstance(effective, dict) or CONFIG_KEY not in effective,
                'language_model_only needs --language-model-only: the startup config pins it off')
        return None
    require(config[CONFIG_KEY] is True, 'pinned language_model_only must be true; off is pinned by leaving it out')
    require(isinstance(effective, dict) and effective.get(CONFIG_KEY) is True,
            'language_model_only differs from the pinned vLLM arguments')
    return True


def record(config):
    """The worker evidence record a startup config expects (RECORD), or None (off)."""
    return dict(RECORD, mm_limits=dict(RECORD['mm_limits'])) if pinned(config) else None


def loaded():
    """The pin of this process's startup config on the HTTP runtime, else None. Startup only."""
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return pinned(load())


def resolved(model_config):
    """vLLM's resolved language_model_only (None: no multimodal config to read)."""
    mm = getattr(model_config, 'multimodal_config', None)
    return None if mm is None else getattr(mm, 'language_model_only', None)


def text_only_limits(model_config):
    """With the flag on: vLLM resolved language-model-only, with limit 0 for every tower modality."""
    mm = getattr(model_config, 'multimodal_config', None)
    require(mm is not None and getattr(mm, 'language_model_only', None) is True,
            'vLLM did not resolve --language-model-only')
    limits = {m: mm.get_limit_per_prompt(m) for m in TOWER_MODALITIES}
    require(all(v == 0 and not isinstance(v, bool) for v in limits.values()),
            'language-model-only needs multimodal limit 0 for %s, vLLM resolved %r'
            % (', '.join(TOWER_MODALITIES), limits))
    return {m: 0 for m in TOWER_MODALITIES}


def tower_names(model):
    """The vision tower's name components: ``visual`` (Qwen3.6); for a Gemma 4 model
    (gemma-s1, D-8) ``vision_tower`` and ``embed_vision`` (model_profile.GEMMA4_TOWERS)."""
    if type(model).__name__.startswith('Gemma4'):
        from .model_profile import GEMMA4_TOWERS
        return GEMMA4_TOWERS
    return (TOWER,)


def tower_tensors(model):
    """(tensors, bytes) of the vision tower the model holds in memory: parameters and
    buffers with a tower name component (``visual`` for Qwen) that are not on the meta device."""
    count = size = 0
    towers = tower_names(model)
    for method in ('named_parameters', 'named_buffers'):
        named = getattr(model, method, None)
        for name, tensor in (named() if callable(named) else ()):
            parts = name.split('.')
            if any(t in parts for t in towers) and getattr(getattr(tensor, 'device', None), 'type', None) != 'meta':
                count += 1
                size += tensor.numel() * tensor.element_size()
    return count, size


def check_engine(vllm_config, pin):
    """compose_engine (HTTP runtime): the engine's resolved config agrees with the pin."""
    model_config = getattr(vllm_config, 'model_config', None)
    if pin is None:
        require(resolved(model_config) is not True,
                'vLLM runs language-model-only but the startup config pins it off')
        return None
    text_only_limits(model_config)
    return True


def worker_check(runner, startup):
    """Worker startup: the record the engine sent, checked once against the pinned config
    and against the loaded runner. Returns RECORD (on) or None (off).

    With the HTTP runtime the worker reads the startup config itself and refuses a
    record that differs; without it (CPU fixtures) only the record's value is checked."""
    sent = startup.get(CONFIG_KEY)
    require(sent is None or sent is True, 'language_model_only worker record must be true or absent')
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') == '1':
        require(loaded() == sent, 'worker language_model_only differs from the pinned startup config')
    model_config = getattr(runner, 'model_config', None)
    if sent is None:
        require(resolved(model_config) is not True,
                'vLLM runs language-model-only but the startup config pins it off')
        return None
    limits = text_only_limits(model_config)
    # The runner's raw model (get_model unwraps vLLM's CUDA-graph wrappers).
    get = getattr(runner, 'get_model', None)
    model = get() if callable(get) else getattr(runner, 'model', None)
    require(model is not None, 'language-model-only needs the loaded model')
    count, size = tower_tensors(model)
    require(count == 0, 'language-model-only loaded %d vision-tower tensors (%d bytes)' % (count, size))
    return dict(mm_limits=limits, vision_tower_tensors=count)
