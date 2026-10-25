"""Gemma 4 stage 1 (gemma-s1 WP1): the model profile, CPU only.

The geometry, stop ids, KV arithmetic and the vision-tower check of
``remote_ra_live_v2.model_profile``. With ``REMOTE_RA_GEMMA4_METADATA`` set to a
directory holding the pinned metadata files, the real config.json and
generation_config.json are checked too.
"""
import copy
import json
import unittest
from types import SimpleNamespace as NS

import gemma4_testing as G
from remote_ra_live_v2 import model_profile as MP


class GeometryTests(unittest.TestCase):
    def test_pinned_geometry(self):
        g = MP.gemma4_geometry(G.config())
        self.assertEqual(g, MP.GEMMA4_GEOMETRY)
        self.assertEqual(g.global_layers, (5, 11, 17, 23, 29, 35, 41, 47, 53, 59))
        self.assertEqual(len(g.sliding_layers), 50)
        self.assertEqual(g.query_layer, 59)
        self.assertIn(g.query_layer, g.global_layers)

    def test_kv_bytes(self):
        # Design 1 (geometry probe): 81,920 B per token over the global layers (1.25 x
        # Qwen's 65,536), 16,384 B per token per sliding layer, 1.25 MiB Pages at P = 16.
        g = MP.GEMMA4_GEOMETRY
        self.assertEqual(g.global_token_bytes, 81920)
        self.assertEqual(g.sliding_layer_token_bytes, 16384)
        self.assertEqual(g.page_bytes(16), 1310720)
        self.assertEqual(g.page_bytes(32), 2621440)
        self.assertEqual(g.sliding_block_bytes(16), 2621440)
        self.assertEqual(g.sliding_block_bytes(32), 5242880)
        # The window state the next query reads: 1,023 tokens of 50 layers.
        self.assertEqual(1023 * g.sliding_layer_token_bytes * len(g.sliding_layers), 838041600)

    def test_sliding_groups_follow_vllm(self):
        groups = MP.GEMMA4_GEOMETRY.sliding_groups()
        sliding = [i for i in range(60) if i % 6 != 5]
        self.assertEqual(groups, tuple(tuple(sliding[i::5]) for i in range(5)))
        self.assertEqual(sorted(i for group in groups for i in group), sliding)
        self.assertEqual({len(group) for group in groups}, {10})

    def test_refusals(self):
        def changed(edit):
            value = copy.deepcopy(G.config())
            edit(value)
            with self.assertRaises(ValueError):
                MP.gemma4_geometry(value)
        changed(lambda c: c.update(model_type='qwen3_5'))
        changed(lambda c: c.update(architectures=['Gemma4ForCausalLM']))
        changed(lambda c: c['text_config'].update(sliding_window=512))
        changed(lambda c: c['text_config'].update(num_global_key_value_heads=8))
        changed(lambda c: c['text_config'].update(global_head_dim=256))
        changed(lambda c: c['text_config'].update(hidden_size=5120))
        changed(lambda c: c['text_config'].update(num_kv_shared_layers=10))
        changed(lambda c: c['text_config'].update(attention_k_eq_v=False))
        changed(lambda c: c['text_config'].update(max_position_embeddings=131072))
        changed(lambda c: c['text_config']['layer_types'].__setitem__(59, 'sliding_attention'))
        changed(lambda c: c['text_config']['layer_types'].__setitem__(0, 'full_attention'))
        changed(lambda c: c['text_config']['layer_types'].__setitem__(3, 'chunked_attention'))
        changed(lambda c: c['text_config'].update(num_hidden_layers=61))
        with self.assertRaises(ValueError):
            MP.gemma4_geometry([])


class StopIdTests(unittest.TestCase):
    def test_generation_config_plus_pad(self):
        ids = MP.gemma4_stop_ids({'eos_token_id': [1, 106, 50]}, 0)
        self.assertEqual(ids, frozenset({0, 1, 50, 106}))
        self.assertEqual(ids, MP.GEMMA4_STOP_IDS)
        self.assertIn(MP.TURN_CLOSE, ids)

    def test_refusals(self):
        for config, pad in (({'eos_token_id': 1}, 0), ({'eos_token_id': [1, 106]}, 0),
                            ({'eos_token_id': [1, 106, 50]}, None), ({}, 0), (None, 0)):
            with self.subTest(config=config, pad=pad), self.assertRaises(ValueError):
                MP.gemma4_stop_ids(config, pad)


class ManifestTests(unittest.TestCase):
    def test_profile_of_a_release(self):
        self.assertEqual(MP.from_manifest({'model_root': '/m'}), MP.QWEN)
        self.assertEqual(MP.from_manifest({'model_profile': MP.GEMMA4}), MP.GEMMA4)
        self.assertTrue(MP.is_gemma4(MP.GEMMA4))
        self.assertFalse(MP.is_gemma4(MP.QWEN))
        for bad in ({'model_profile': MP.QWEN}, {'model_profile': 'llama'}, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MP.from_manifest(bad)


class Tensor:
    def __init__(self, device, n):
        self.device, self.n = NS(type=device), n

    def numel(self):
        return self.n

    def element_size(self):
        return 2


class VisionTowerTests(unittest.TestCase):
    """D-8: the worker's text-only check (language_model_only.tower_tensors) looks for
    Gemma's tower names on a Gemma 4 model and for Qwen's ``visual`` otherwise."""

    def model(self, cls, params):
        return type(cls, (), dict(named_parameters=lambda self: iter(params),
                                  named_buffers=lambda self: iter(())))()

    def test_counts_held_tower_tensors_only(self):
        from remote_ra_live_v2 import language_model_only as T
        params = [('language_model.model.layers.0.self_attn.qkv_proj.weight', Tensor('cuda', 10)),
                  ('vision_tower.encoder.layers.0.mlp.fc1.weight', Tensor('meta', 7)),
                  ('model.embed_vision.embedding_projection.weight', Tensor('meta', 3))]
        gemma = self.model('Gemma4ForConditionalGeneration', params)
        self.assertEqual(T.tower_names(gemma), MP.GEMMA4_TOWERS)
        self.assertEqual(T.tower_tensors(gemma), (0, 0))
        params.append(('vision_tower.encoder.layers.0.mlp.fc2.weight', Tensor('cuda', 5)))
        params.append(('model.embed_vision.embedding_projection.weight', Tensor('cpu', 4)))
        self.assertEqual(T.tower_tensors(gemma), (2, 18))
        # A Qwen model is checked for its own tower only, as before.
        qwen = self.model('Qwen3_5ForConditionalGeneration', params)
        self.assertEqual(T.tower_names(qwen), ('visual',))
        self.assertEqual(T.tower_tensors(qwen), (0, 0))
        # A name that merely contains the word is not a tower component.
        params[:] = [('language_model.vision_tower_like.weight', Tensor('cuda', 5))]
        self.assertEqual(T.tower_tensors(gemma), (0, 0))


@unittest.skipUnless(G.metadata_dir() is not None, G.ENV + ' (pinned Gemma 4 metadata) not set')
class PinnedMetadataTests(unittest.TestCase):
    def test_real_config_and_generation_config(self):
        root = G.metadata_dir()
        config = json.loads((root / 'config.json').read_bytes())
        self.assertEqual(MP.gemma4_geometry(config), MP.GEMMA4_GEOMETRY)
        generation = json.loads((root / 'generation_config.json').read_bytes())
        self.assertEqual(MP.gemma4_stop_ids(generation, config['text_config']['pad_token_id']),
                         MP.GEMMA4_STOP_IDS)
        self.assertEqual(tuple(generation['eos_token_id']), MP.GEMMA4_GENERATION_EOS)
        # The synthetic fixture carries the real values of every checked field.
        text, mine = config['text_config'], G.config()['text_config']
        self.assertEqual({k: text[k] for k in mine}, mine)


if __name__ == '__main__':
    unittest.main()
