"""Reply bound and context limit: server settings pinned at startup, checked once.

``serve.py --max-tokens {1536,4096}`` pins the server's reply bound, the new tokens
one request may generate. It is a server setting, not part of an agent protocol:
the frontend's live controller and the engine's private endpoint hold it, and each
protocol enforces it (today the Mini-SWE-Agent XML protocol: the route fixes it on
every request, ``mswea_route.prepare_request``, and EngineCore refuses any other
value before allocation, ``mswea_protocol.check_sampling``). The capacity gate and
the launcher's never-deadlock arithmetic reserve ceil(max_tokens / P) Pages for a
reply. The server still decodes greedily and ignores the client's sampling fields.

1536 (the default) is the dram-backed-kv PoC's value and the qualified profile,
byte for byte: startup.json has no ``max_tokens`` key, and the worker startup
record, the worker evidence and ready.json gain nothing. 4096 adds ``max_tokens``
to startup.json (covered by REMOTE_RA_HTTP_CONFIG_SHA256 like every key). vLLM caps
a reply at max_model_len minus the prompt, so a prompt holds at most max_model_len
- max_tokens tokens (a longer prompt gets a smaller cap, which check_sampling
refuses).

``serve.py --max-model-len {65536,131072,262144}`` pins the context limit, prompt
plus reply. 65536 is the default and the qualified profile; 131072 and 262144
change vLLM's --max-model-len and effective_server_config.max_model_len only
(262144 is the model's max_position_embeddings with its default rope: no YaRN,
no other pin). What follows from it is sized at startup from vLLM's own
max_model_len: the Page catalog bound, the Mamba/attention rows and the worker's
pinned Page arena (room for every running session at the limit, S x (ceil(L / P)
+ 1) Pages, unless ``serve.py --dram-arena-gib`` pins one smaller arena for all
sessions as ``dram_arena_pages``; a capture beyond it fails with
DRAM_ARENA_EXHAUSTED, nothing is spilled or evicted). The launcher refuses a KV
budget that cannot hold one session's turn at the limit and a server whose
worst-case pinned host memory exceeds its stated budget (serve.py
PINNED_HOST_GIB).

One request's byte and token caps (the XML source, the core request, the
authenticated frame, EngineCore's registered-add message and the prompt ids)
scale with the pinned limit (``request_caps``). At 65536 and 131072 they are
the values the sources carry, byte for byte, and nothing is set; at 262144
``pin_request_caps`` sets the doubled caps once at startup, in the frontend
(``compose_frontend``) and the engine (``compose_engine``).

Checked once at startup, never per request or step:

- ``remote_ra_http.config.load``: the key is absent (the default) or a non-default
  choice, below the pinned context limit; max_model_len is one of the choices; a
  pinned arena needs the decoupled layout and lies between one session and all;
- frontend: vLLM's parsed --max-model-len equals the pinned one
  (``effective_server_config``);
- frontend: ``compose_frontend`` gives the live controller the pinned bound;
- frontend and engine: ``pin_request_caps`` installs the request caps of a
  262144-token server (nothing at 65536 and 131072);
- engine: ``SplitPrivateEndpoint`` takes the pinned bound; ``compose_engine``
  sends the non-default record {max_model_len, max_tokens} to the worker in its
  startup record, compares the worker's echo and checks vLLM's resolved
  max_model_len;
- worker (``attach_loaded_worker``): the record must match its runner's
  max_model_len; it is echoed in the worker evidence, which the engine
  (``worker_attached``) and the frontend (``write_ready``) compare with the
  startup config, and ready.json records it.
"""
from __future__ import annotations

from .contract import require

DEFAULT_MAX_TOKENS = 1536      # the PoC's reply bound (mswea_protocol.MAX_TOKENS)
MAX_TOKENS_CHOICES = (1536, 4096)   # also written out in serve.py's --max-tokens choices
CONFIG_KEY = 'max_tokens'      # startup config; absent: DEFAULT_MAX_TOKENS
RECORD_KEY = 'context'         # worker startup record, worker evidence, ready.json; absent: the defaults
DEFAULT_MAX_MODEL_LEN = 65536  # the qualified profile (serve.py MAX_MODEL_LEN)
MAX_MODEL_LENS = (65536, 131072, 262144)   # also written out in serve.py's --max-model-len choices
ARENA_KEY = 'dram_arena_pages' # startup config and worker startup record (serve.py --dram-arena-gib)
# One request's caps as the sources carry them, the values at 65536 and 131072:
# runtime_renderer.MAX_SOURCE_BYTES, registered_add_protocol.MAX_CORE_BYTES and
# MAX_WIRE_BYTES, EngineCore's registered-add message bound (vllm/v1/engine/core.py)
# and the prompt ids (registered_add_protocol.MAX_TOKENS, runtime_renderer.MAX_PROMPT_TOKENS).
SOURCE_CAPS = dict(source_bytes=1 << 20, core_bytes=1 << 20, wire_bytes=4 << 20,
                   input_bytes=5 << 20, prompt_tokens=131072)
# Where pin_request_caps sets them: (module, attribute, cap). The engine's message
# bound is an attribute of the EngineCore (ENGINE_INPUT_ATTRIBUTE).
CAP_TARGETS = (('runtime_renderer', 'MAX_SOURCE_BYTES', 'source_bytes'),
               ('runtime_renderer', 'MAX_PROMPT_TOKENS', 'prompt_tokens'),
               ('registered_add_protocol', 'MAX_CORE_BYTES', 'core_bytes'),
               ('registered_add_protocol', 'MAX_WIRE_BYTES', 'wire_bytes'),
               ('registered_add_protocol', 'MAX_TOKENS', 'prompt_tokens'))
ENGINE_INPUT_ATTRIBUTE = '_remote_ra_registered_frame_bytes'


def max_tokens(config):
    """The reply bound pinned by a startup config (None, or no key: the default)."""
    if config is None or CONFIG_KEY not in config:
        return DEFAULT_MAX_TOKENS
    value = config[CONFIG_KEY]
    require(type(value) is int and value in MAX_TOKENS_CHOICES and value != DEFAULT_MAX_TOKENS,
            'pinned max_tokens must be one of %s; the default %d is pinned by leaving the key out'
            % (', '.join(str(v) for v in MAX_TOKENS_CHOICES if v != DEFAULT_MAX_TOKENS), DEFAULT_MAX_TOKENS))
    return value


def max_model_len(config):
    """The context limit pinned by a startup config's effective_server_config
    (None, or no effective config: the default)."""
    effective = None if config is None else config.get('effective_server_config')
    if not isinstance(effective, dict) or 'max_model_len' not in effective:
        return DEFAULT_MAX_MODEL_LEN
    value = effective['max_model_len']
    require(type(value) is int and value in MAX_MODEL_LENS,
            'pinned max_model_len must be one of %s' % ', '.join(str(v) for v in MAX_MODEL_LENS))
    return value


def check_config(config):
    """Startup config check (remote_ra_http.config.load): both values valid and a
    reply bound below the context limit. Returns (max_tokens, max_model_len)."""
    T, L = max_tokens(config), max_model_len(config)
    require(T < L, 'max_tokens %d must be below max_model_len %d' % (T, L))
    return T, L


def session_pages(page_tokens, length):
    """Host Pages of one session at the context limit: every Page plus the tail."""
    return -(-length // page_tokens) + 1


def arena_pages(config, sessions):
    """The pinned Page arena capacity a startup config pins (``dram_arena_pages``,
    serve.py --dram-arena-gib), or None: room for every running session at the
    context limit (worker_composition.arena_pages). Decoupled layout only; at least
    one session at the limit, less than every session."""
    if config is None or ARENA_KEY not in config:
        return None
    value, P, L = config[ARENA_KEY], config.get('page_tokens'), max_model_len(config)
    require(config.get('kv_layout') is not None, 'dram_arena_pages needs the decoupled KV layout')
    require(type(P) is int and P > 0 and type(sessions) is int and sessions > 0, 'pinned geometry required')
    one = session_pages(P, L)
    require(type(value) is int and one <= value < sessions * one,
            'dram_arena_pages must hold one session at the %d-token limit (%d Pages) and less than %d '
            'sessions (%d Pages)' % (L, one, sessions, sessions * one))
    return value


def request_caps(length):
    """One request's caps at the context limit ``length`` (one of MAX_MODEL_LENS):

    - ``source_bytes`` max(1 MiB, 8 B x L): the XML conversation JSON (frontend and
      engine) and every encoded message text;
    - ``core_bytes`` max(1 MiB, 8 B x L): the msgpack EngineCoreRequest, whose
      prompt ids take 1 to 5 bytes each;
    - ``wire_bytes`` 2 x (source + core): the authenticated frame, the base64 of a
      payload that holds the base64 core request and the source;
    - ``input_bytes`` wire + 1 MiB: EngineCore's bound on the msgpack
      registered-add message (its input thread);
    - ``prompt_tokens`` max(131072, L): the prompt ids of one request.

    At 65536 and 131072 this is SOURCE_CAPS; at 262144 every cap doubles (2 MiB,
    2 MiB, 8 MiB, 9 MiB, 262144)."""
    require(type(length) is int and length in MAX_MODEL_LENS,
            'context limit must be one of %s' % ', '.join(str(v) for v in MAX_MODEL_LENS))
    source = max(SOURCE_CAPS['source_bytes'], 8 * length)
    core = max(SOURCE_CAPS['core_bytes'], 8 * length)
    wire = 2 * (source + core)
    return dict(source_bytes=source, core_bytes=core, wire_bytes=wire, input_bytes=wire + (1 << 20),
                prompt_tokens=max(SOURCE_CAPS['prompt_tokens'], length))


def pin_request_caps(config, core=None):
    """Startup only (``compose_frontend``; ``compose_engine`` with its EngineCore
    ``core``): install the request caps of the pinned context limit, once. At 65536
    and 131072 (or without a config) they are the sources' own values: nothing is
    imported or set and None is returned. At 262144 the module caps (CAP_TARGETS)
    are set, a repeated pin must agree, and the engine's message bound becomes the
    attribute ENGINE_INPUT_ATTRIBUTE of ``core``. Returns the caps it set, or None."""
    if config is None:
        return None
    caps = request_caps(max_model_len(config))
    if caps == SOURCE_CAPS:
        return None
    import importlib
    modules = {name: importlib.import_module(name) for name, _, _ in CAP_TARGETS}
    for name, attribute, key in CAP_TARGETS:
        require(getattr(modules[name], attribute) in (SOURCE_CAPS[key], caps[key]),
                '%s.%s is pinned for another context limit' % (name, attribute))
    if core is not None:
        require(getattr(core, ENGINE_INPUT_ATTRIBUTE, None) in (None, caps['input_bytes']),
                'the engine registered-add bound is pinned for another context limit')
    for name, attribute, key in CAP_TARGETS:
        setattr(modules[name], attribute, caps[key])
    if core is not None:
        setattr(core, ENGINE_INPUT_ATTRIBUTE, caps['input_bytes'])
    return caps


def record(config):
    """The worker/evidence record {max_model_len, max_tokens}, or None at the defaults."""
    T, L = check_config(config)
    if (T, L) == (DEFAULT_MAX_TOKENS, DEFAULT_MAX_MODEL_LEN):
        return None
    return dict(max_model_len=L, max_tokens=T)


def loaded():
    """This process's startup config on the HTTP runtime, else None. Startup only."""
    import os
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None
    from remote_ra_http.config import load
    return load()


def engine_record():
    """compose_engine: the record for the worker startup record, or None (the
    defaults, or no HTTP runtime)."""
    config = loaded()
    return None if config is None else record(config)


def engine_arena():
    """compose_engine: the pinned arena capacity for the worker startup record, or
    None (room for every session; no HTTP runtime). load() validated it."""
    config = loaded()
    return None if config is None else config.get(ARENA_KEY)


def check_bound(value):
    """A reply bound handed to a frontend or engine owner at startup."""
    require(type(value) is int and value in MAX_TOKENS_CHOICES,
            'reply bound must be one of %s' % ', '.join(str(v) for v in MAX_TOKENS_CHOICES))
    return value


def runner_max_model_len(runner):
    length = getattr(runner, 'max_model_len', None)
    if length is None:
        length = runner.model_config.max_model_len
    return int(length)


def check_engine(vllm_config, config):
    """Engine startup: vLLM's resolved context limit is the pinned one."""
    if config is None:
        return None
    L = max_model_len(config)
    actual = getattr(getattr(vllm_config, 'model_config', None), 'max_model_len', None)
    require(actual == L, 'engine max_model_len %r differs from the pinned %d' % (actual, L))
    return L


def check_worker(runner, value):
    """Worker startup: the record sent by compose_engine (None: the defaults, no check)."""
    if value is None:
        return None
    require(type(value) is dict and set(value) == {'max_model_len', 'max_tokens'},
            'context record must be {max_model_len, max_tokens}')
    T, L = value['max_tokens'], value['max_model_len']
    require(type(T) is int and T in MAX_TOKENS_CHOICES and type(L) is int and L in MAX_MODEL_LENS
            and T < L and (T, L) != (DEFAULT_MAX_TOKENS, DEFAULT_MAX_MODEL_LEN),
            'context record must hold pinned, non-default values')
    actual = runner_max_model_len(runner)
    require(actual == L, 'worker max_model_len %d differs from the pinned %d' % (actual, L))
    return dict(max_model_len=L, max_tokens=T)
