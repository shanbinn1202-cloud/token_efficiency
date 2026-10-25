"""Interface I0 (exp-decouple-pages): remote_ra_live_v2/kv_layout.py, pure Python."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'source-overlay'))

from remote_ra_live_v2 import kv_layout as K
from remote_ra_live_v2.kv_layout import KvLayout


class KvLayoutTests(unittest.TestCase):
    def test_legacy_is_the_shared_784_pool(self):
        layout = K.legacy()
        self.assertEqual(layout, KvLayout(784, 784, 0, False))
        self.assertFalse(layout.decoupled)
        self.assertEqual((K.LAYOUT_KEY, K.LEGACY_PAGE), ('remote_ra_kv_layout', 784))
        self.assertIsNone(K.to_additional_config(layout))

    def test_default_mamba_tokens(self):
        self.assertEqual([K.default_mamba_tokens(p) for p in (16, 32, 64, 784)], [784, 800, 832, 784])
        for bad in (0, 8, 24, -16, 16.0, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                K.default_mamba_tokens(bad)

    def test_for_server(self):
        self.assertEqual(K.for_server(784, 1), K.legacy())
        self.assertEqual(K.for_server(784, 8, False), K.legacy())
        self.assertEqual(K.for_server(16, 1), KvLayout(16, 784, 3, True))
        self.assertEqual(K.for_server(16, 8), KvLayout(16, 784, 17, True))
        self.assertEqual(K.for_server(32, 4), KvLayout(32, 800, 9, True))
        self.assertEqual(K.for_server(64, 4), KvLayout(64, 832, 9, True))
        # P = 784 + decoupled is the pool-split validation mode.
        self.assertEqual(K.for_server(784, 2, True), KvLayout(784, 784, 5, True))
        for args in ((16, 1, False), (16, 0), (16, -1), (16, True), (48 + 1, 1), (8, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                K.for_server(*args)

    def test_validation(self):
        for args in ((16, 784, 3, True), (784, 784, 3, True), (64, 832, 3, True),
                     (16, 16, 3, True), (16, 16384, 3, True)):
            KvLayout(*args)
        bad = [(0, 784, 3, True), (8, 784, 3, True), (24, 792, 3, True),   # P % 16
               (32, 784, 3, True),                                          # M % P
               (16, 0, 3, True), (16, 16400, 3, True), (16, -16, 3, True),  # 0 < M <= 16384
               (16, 784, 2, True), (16, 784, 0, True),                      # decoupled => K >= 3
               (784, 784, 3, False), (16, 784, 0, False), (784, 1568, 0, False),
               (784.0, 784, 0, False), (784, 784, 0, 0), (True, 784, 3, True)]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                KvLayout(*args)

    def test_frozen(self):
        with self.assertRaises(Exception):
            K.legacy().page_tokens = 16

    def test_additional_config_round_trip(self):
        for layout in (K.for_server(16, 1), K.for_server(64, 8), K.for_server(784, 1, True)):
            with self.subTest(layout=layout):
                extra = K.to_additional_config(layout)
                self.assertEqual(extra, {'remote_ra_kv_layout': {'page_tokens': layout.page_tokens,
                    'mamba_tokens': layout.mamba_tokens, 'mamba_slots': layout.mamba_slots}})
                # Survives the --additional-config JSON argv round trip.
                extra = json.loads(json.dumps(extra))
                self.assertEqual(K.from_vllm_config(NS(additional_config=extra)), layout)
                # Unrelated keys are ignored.
                extra['other'] = 1
                self.assertEqual(K.from_vllm_config(NS(additional_config=extra)), layout)

    def test_from_vllm_config_legacy_and_errors(self):
        for config in (NS(additional_config={}), NS(additional_config=None), NS(),
                       NS(additional_config={'other': 1}), NS(additional_config=object())):
            self.assertEqual(K.from_vllm_config(config), K.legacy())
        good = {'page_tokens': 16, 'mamba_tokens': 784, 'mamba_slots': 3}
        for value in ({}, [], None, dict(good, extra=1), {'page_tokens': 16, 'mamba_tokens': 784},
                      dict(good, page_tokens='16'), dict(good, mamba_slots=3.0),
                      dict(good, mamba_slots=True), dict(good, mamba_slots=2),
                      dict(good, mamba_tokens=792), dict(good, page_tokens=32)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                K.from_vllm_config(NS(additional_config={K.LAYOUT_KEY: value}))

    def test_mamba_pool_blocks(self):
        self.assertEqual(K.mamba_pool_blocks(K.for_server(16, 1), 3), 10)
        self.assertEqual(K.mamba_pool_blocks(K.for_server(16, 4), 3), 28)
        self.assertEqual(K.mamba_pool_blocks(K.for_server(16, 8), 3), 52)
        with self.assertRaises(ValueError):
            K.mamba_pool_blocks(K.legacy(), 3)
        for groups in (0, -1, 3.0, True):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                K.mamba_pool_blocks(K.for_server(16, 1), groups)

    def test_mamba_row_len_and_index(self):
        for layout in (K.legacy(), K.for_server(16, 1), K.for_server(784, 1, True)):
            self.assertEqual(layout.mamba_tokens, 784)
            for end, row, index in ((1, 1, 0), (783, 1, 0), (784, 1, 0), (785, 2, 1),
                                    (1568, 2, 1), (1569, 3, 2), (65536, 84, 83)):
                with self.subTest(layout=layout, end=end):
                    self.assertEqual(K.mamba_row_len(end, layout), row)
                    self.assertEqual(K.mamba_index(end, layout), index)
        self.assertEqual(K.mamba_row_len(0, K.legacy()), 0)
        p64 = K.for_server(64, 1)
        self.assertEqual((K.mamba_row_len(832, p64), K.mamba_row_len(833, p64)), (1, 2))
        self.assertEqual((K.mamba_index(832, p64), K.mamba_index(833, p64)), (0, 1))
        for end in (-1, 1.0, True):
            with self.subTest(end=end), self.assertRaises(ValueError):
                K.mamba_row_len(end, p64)
        for end in (0, -1):
            with self.subTest(end=end), self.assertRaises(ValueError):
                K.mamba_index(end, p64)


if __name__ == '__main__':
    unittest.main()
