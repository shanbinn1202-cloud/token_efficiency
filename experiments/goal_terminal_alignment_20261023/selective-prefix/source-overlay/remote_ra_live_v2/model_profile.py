"""Model profiles: Qwen3.6-27B (the runtime as it was) and Gemma 4 31B IT (stage 1).

Pure Python, no runtime or vLLM imports, so the launcher, the installer and the
vLLM overlay can import it lazily. Everything here is evaluated at startup only.

* ``QWEN`` is the profile of every release without a ``model_profile`` key in its
  manifest. Its name is never written anywhere: the vLLM command, the startup
  config, the pins and the evidence of a Qwen server are unchanged.
* ``GEMMA4`` (stage 1, not GPU-validated) is google/gemma-4-31B-it at the pinned
  revision. Its 60 decoder layers are 50 sliding-window layers (window 1024, 16 KV
  heads x 256) and 10 global full-attention layers (5, 11, ..., 59; 4 KV heads x
  512, K and V both cached). Runtime Pages cover the global layers only; the
  sliding layers always attend to the exact last 1,023 tokens plus the current one,
  as in stock vLLM, and their window is outside the active-token budget (D-1).

The model's metadata files are not copied into the overlay. The installer pins
them by SHA-256 (``MODEL_PINS``; ``INSTALL_RELEASE.json`` ``model_profiles``), and
the runtime reads them from the pinned model directory.
"""
from __future__ import annotations

from dataclasses import dataclass

QWEN = 'qwen3.6-27b'
GEMMA4 = 'gemma4-31b-it'
PROFILES = (QWEN, GEMMA4)
MANIFEST_KEY = 'model_profile'      # runtime manifest and startup config key (Gemma only)

# ---------------------------------------------------------------- Gemma 4 31B IT

GEMMA4_REPOSITORY = 'google/gemma-4-31B-it'
GEMMA4_REVISION = '842da3794eaa0b77d5f08bae87a17459d91ff475'
# SHA-256 of the metadata files of that revision, as the stage-0 model gate (job
# 3364469) checked them in the checkpoint directory.
GEMMA4_MODEL_PINS = {
    'chat_template.jinja': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
    'config.json': 'e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1',
    'generation_config.json': 'd4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de',
    'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'tokenizer_config.json': '9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633',
}

# Special token ids of the pinned tokenizer (tokenizer.json added tokens).
PAD, EOS, BOS = 0, 1, 2
TOOL_RESPONSE = 50      # <|tool_response>
CHANNEL_OPEN, CHANNEL_CLOSE = 100, 101   # <|channel>, <channel|>
TURN_OPEN, TURN_CLOSE = 105, 106         # <|turn>, <turn|>
# Stop ids: pad plus generation_config.json eos_token_id [1, 106, 50]. The
# tokenizer's own EOS and pad (the Qwen rule) would miss <turn|>, the normal end
# of a model turn (D-11).
GEMMA4_STOP_IDS = frozenset((PAD, EOS, TOOL_RESPONSE, TURN_CLOSE))
GEMMA4_GENERATION_EOS = (EOS, TURN_CLOSE, TOOL_RESPONSE)


@dataclass(frozen=True)
class Geometry:
    """KV geometry of a Gemma-4-style model (sliding and global attention layers)."""
    layers: int
    global_layers: tuple[int, ...]
    hidden_size: int
    sliding_window: int
    sliding_kv_heads: int
    sliding_head_size: int
    global_kv_heads: int
    global_head_size: int
    dtype_bytes: int = 2            # bf16 KV cache

    @property
    def sliding_layers(self):
        return tuple(i for i in range(self.layers) if i not in self.global_layers)

    @property
    def query_layer(self):
        """The selection query's layer: the last decoder layer (Selector V2 contract)."""
        return self.layers - 1

    @property
    def global_token_bytes(self):
        """KV bytes of one token over every global layer (K and V both cached)."""
        return 2 * self.global_kv_heads * self.global_head_size * self.dtype_bytes * len(self.global_layers)

    @property
    def sliding_layer_token_bytes(self):
        """KV bytes of one token in one sliding layer."""
        return 2 * self.sliding_kv_heads * self.sliding_head_size * self.dtype_bytes

    def sliding_groups(self):
        """vLLM's sliding KV cache groups: the smaller layer count (the global
        layers) is the group size, layers[i::n] (kv_cache_utils
        _get_kv_cache_groups_uniform_page_size)."""
        sliding, size = self.sliding_layers, len(self.global_layers)
        if len(sliding) % size:
            raise ValueError('sliding layers %d are not a multiple of the global layers %d'
                             % (len(sliding), size))
        count = len(sliding) // size
        return tuple(sliding[i::count] for i in range(count))

    def page_bytes(self, page_tokens):
        """One runtime Page: ``page_tokens`` tokens of every global layer."""
        return page_tokens * self.global_token_bytes

    def sliding_block_bytes(self, block_tokens):
        """One block id of pool 1: ``block_tokens`` tokens of the layers of one sliding group."""
        return block_tokens * self.sliding_layer_token_bytes * len(self.global_layers)


GEMMA4_GEOMETRY = Geometry(layers=60, global_layers=tuple(range(5, 60, 6)), hidden_size=5376,
                           sliding_window=1024, sliding_kv_heads=16, sliding_head_size=256,
                           global_kv_heads=4, global_head_size=512)

# Launcher choices of the Gemma profile (D-2, D-4, D-7, D-8).
GEMMA4_PAGE_TOKENS = (16, 32)      # P = 16 (D-2), P = 32 the fallback
GEMMA4_FORWARD_TOKENS = 4096       # one native forward (max_num_batched_tokens, D-4)
# goal_query (gemma-goal): the largest Goal forward, the window layout's Goal transient
# (kv_layout.WindowLayout.goal_tokens). The Goal forward is a separate cold forward of the
# <retrieval_query> text that runs after phase 1 while phase 1's fresh sliding blocks are
# still held, so pool 1 reserves ceil(G / P) more blocks per sliding group for it (1,600 MiB
# at either Page size). Qwen's Goal forwards in the 16-session SWE-bench run (job 3363871,
# 32 tasks) took at most 1,164 tokens; a longer Goal fails its turn (PHASE1_TOO_LONG), as
# a Goal longer than one native forward does on Qwen. 2,048 is the largest power of two
# that keeps 16 goal_query sessions at 131,072 tokens within --kv-cache-gib 24.
GEMMA4_GOAL_TOKENS = 2048
# Chunked phase 1 (gemma-goal, design WP10): a phase 1 longer than one native forward
# (an observation of more than about 8K tokens: digit-dense output at the 10,000-character
# cap is about 10K) runs as forwards of at most 4,096 tokens on the same lease; between
# them the fresh sliding blocks that left the window are reused, so pool 1 holds at most
# ceil((1,023 + 4,096) / P) + 1 fresh blocks per sliding group for it (321 at P = 16,
# within the one forward's transient of 512).
GEMMA4_PHASE1_CHUNK_TOKENS = 4096
GEMMA4_ATTENTION_BACKEND = 'TRITON_ATTN'   # D-7: FA3 has no 512-wide heads; FA4 is not validated
GEMMA4_PROTOCOL = 'mswea_xml_gemma4_v1'
# Vision tower name components (vLLM and Transformers Gemma4ForConditionalGeneration):
# with --language-model-only none may be held off the meta device (D-8;
# the worker's text-only check, tower_tensors, looks once at startup).
GEMMA4_TOWERS = ('vision_tower', 'embed_vision')

_LAYER_TYPES = {'sliding_attention', 'full_attention'}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def gemma4_geometry(config):
    """The Geometry of a parsed Gemma 4 ``config.json`` (top level with ``text_config``),
    refused unless it is exactly the pinned model's: the profile's KV arithmetic, the
    selection query layer and the two-pool layout depend on every value checked here."""
    _need(type(config) is dict and config.get('model_type') == 'gemma4'
          and config.get('architectures') == ['Gemma4ForConditionalGeneration'],
          'Gemma 4 config.json (Gemma4ForConditionalGeneration) required')
    text = config.get('text_config')
    _need(type(text) is dict and text.get('model_type') == 'gemma4_text', 'Gemma 4 text_config required')
    types = text.get('layer_types')
    _need(type(types) is list and set(types) <= _LAYER_TYPES and len(types) == text.get('num_hidden_layers'),
          'layer_types must name every decoder layer sliding_attention or full_attention')
    geometry = Geometry(layers=len(types),
                        global_layers=tuple(i for i, t in enumerate(types) if t == 'full_attention'),
                        hidden_size=text.get('hidden_size'), sliding_window=text.get('sliding_window'),
                        sliding_kv_heads=text.get('num_key_value_heads'), sliding_head_size=text.get('head_dim'),
                        global_kv_heads=text.get('num_global_key_value_heads'),
                        global_head_size=text.get('global_head_dim'))
    _need(geometry == GEMMA4_GEOMETRY, 'Gemma 4 geometry differs from the pinned profile: %r' % (geometry,))
    # The query layer is a global layer and nothing shares KV across layers.
    _need(types[-1] == 'full_attention' and text.get('num_kv_shared_layers') == 0
          and text.get('attention_k_eq_v') is True and text.get('enable_moe_block') is False
          and text.get('hidden_size_per_layer_input') == 0 and text.get('dtype') == 'bfloat16',
          'Gemma 4 text_config has a feature this profile does not cover')
    _need(text.get('max_position_embeddings') == 262144, 'Gemma 4 context differs from 262144')
    return geometry


def gemma4_stop_ids(generation_config, pad_token_id):
    """The stop set of the Gemma profile from the pinned generation_config.json: its
    eos_token_id list plus the tokenizer's pad id."""
    eos = generation_config.get('eos_token_id') if type(generation_config) is dict else None
    _need(type(eos) is list and eos and all(type(t) is int and t >= 0 for t in eos),
          'generation_config.json eos_token_id must be a list of ids')
    _need(type(pad_token_id) is int and pad_token_id >= 0, 'tokenizer pad id required')
    ids = frozenset(eos) | {pad_token_id}
    _need(ids == GEMMA4_STOP_IDS, 'Gemma 4 stop ids differ from the pinned profile: %r' % sorted(ids))
    return ids


def from_manifest(manifest):
    """The profile name of an installed release: QWEN without the key (every earlier release)."""
    value = manifest.get(MANIFEST_KEY, QWEN) if type(manifest) is dict else None
    _need(value in PROFILES and (value != QWEN or MANIFEST_KEY not in manifest),
          'unknown model profile %r (the Qwen profile is never written)' % (value,))
    return value


def is_gemma4(name):
    return name == GEMMA4
