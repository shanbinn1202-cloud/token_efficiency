from __future__ import annotations

from dataclasses import dataclass
import unittest

from dram_kv.integrations.qwen35 import install_qwen35_logical_rope


class ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


class FakeRotary:
    def __init__(self):
        self.calls = []

    def __call__(self, hidden, positions):
        self.calls.append((hidden, positions))
        return ("logical-cos", "logical-sin")


class FakeAttention:
    def __init__(self):
        self.calls = []

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        self.calls.append(
            (position_embeddings, attention_mask, past_key_values, dict(kwargs))
        )
        return "output"


@dataclass
class Layer:
    self_attn: object | None = None


@dataclass
class Body:
    layers: list
    rotary_emb: object


@dataclass
class Model:
    model: object


class Qwen35IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.rotary = FakeRotary()
        self.attention = FakeAttention()
        self.model = Model(Body([Layer(), Layer(self.attention)], self.rotary))

    def test_logical_positions_replace_rope_only(self):
        self.assertEqual(install_qwen35_logical_rope(self.model), 1)
        hidden = ShapeOnly((1, 3, 8))
        logical = ShapeOnly((1, 3))
        result = self.attention.forward(
            hidden,
            ("physical-cos", "physical-sin"),
            attention_mask="physical-mask",
            past_key_values="cache",
            dram_kv_logical_position_ids=logical,
            untouched=True,
        )
        self.assertEqual(result, "output")
        self.assertEqual(self.rotary.calls, [(hidden, logical)])
        self.assertEqual(
            self.attention.calls,
            [(('logical-cos', 'logical-sin'), "physical-mask", "cache", {"untouched": True})],
        )

    def test_absent_logical_positions_preserve_stock_rope(self):
        install_qwen35_logical_rope(self.model)
        self.attention.forward(ShapeOnly((1, 2, 8)), ("stock-cos", "stock-sin"))
        self.assertEqual(self.rotary.calls, [])
        self.assertEqual(
            self.attention.calls[0][0], ("stock-cos", "stock-sin")
        )

    def test_install_is_idempotent_and_keyword_is_stable(self):
        self.assertEqual(install_qwen35_logical_rope(self.model), 1)
        self.assertEqual(install_qwen35_logical_rope(self.model), 0)
        with self.assertRaisesRegex(RuntimeError, "different"):
            install_qwen35_logical_rope(self.model, keyword="other_positions")

    def test_suffix_length_mismatch_is_rejected(self):
        install_qwen35_logical_rope(self.model)
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.attention.forward(
                ShapeOnly((1, 2, 8)),
                ("physical-cos", "physical-sin"),
                dram_kv_logical_position_ids=ShapeOnly((1, 3)),
            )


if __name__ == "__main__":
    unittest.main()
