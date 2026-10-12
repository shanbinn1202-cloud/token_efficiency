from __future__ import annotations

from dataclasses import dataclass
import unittest

from dram_kv.runtime.qwen import (
    QwenKVPageAdapter,
    assemble_pages,
    extract_page,
    extract_suffix_page,
    full_attention_layers,
    write_active_cache,
)


class FakeTensor:
    element_bytes = 2

    def __init__(self, data, *, device="gpu", dtype="bf16", pinned=False):
        self.data = [[list(tokens) for tokens in heads] for heads in data]
        self.device = device
        self.dtype = dtype
        self.pinned = pinned
        self.shape = (len(self.data), len(self.data[0]), len(self.data[0][0]), 1)
        self.ndim = 4

    @property
    def nbytes(self):
        result = self.element_bytes
        for size in self.shape:
            result *= size
        return result

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.data, device=self.device, dtype=self.dtype, pinned=self.pinned)

    def to(self, device, copy=False):
        if not copy and device == self.device:
            return self
        return FakeTensor(self.data, device=str(device), dtype=self.dtype, pinned=False)

    def pin_memory(self):
        return FakeTensor(self.data, device=self.device, dtype=self.dtype, pinned=True)

    def __getitem__(self, indices):
        seq = indices[-2]
        return FakeTensor(
            [[head[seq] for head in batch] for batch in self.data],
            device=self.device,
            dtype=self.dtype,
            pinned=self.pinned,
        )


@dataclass
class FullLayer:
    keys: FakeTensor
    values: FakeTensor
    is_initialized: bool = True


@dataclass
class RecurrentLayer:
    recurrent_states: object
    keys: object = None
    values: object = None


@dataclass
class Cache:
    layers: list


def tensor(tokens, *, offset=0, device="gpu", dtype="bf16"):
    return FakeTensor([[[value + offset for value in tokens]]], device=device, dtype=dtype)


def fake_cat(tensors, dim):
    assert dim == -2
    values = []
    for item in tensors:
        values.extend(item.data[0][0])
    return tensor(values, device=tensors[0].device, dtype=tensors[0].dtype)


class QwenRuntimeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.recurrent = {"state": 7}
        self.cache = Cache([
            RecurrentLayer(self.recurrent),
            FullLayer(tensor(range(6), offset=10), tensor(range(6), offset=20)),
            RecurrentLayer({"state": 8}),
            FullLayer(tensor(range(6), offset=30), tensor(range(6), offset=40)),
        ])

    def test_extract_transfer_assemble_and_write_roundtrip(self):
        first = extract_page(self.cache, 100, 102, cache_logical_start=100)
        second = extract_page(self.cache, 104, 106, cache_logical_start=100)
        self.assertEqual(first.layer_ids, (1, 3))
        self.assertEqual(first.logical_positions, (100, 101))

        adapter = QwenKVPageAdapter("gpu")
        # 2 layers * K/V * 2 tokens * 2 bytes.
        self.assertEqual(adapter.nbytes(first), 16)
        cpu = adapter.to_cpu(first, pin=True)
        self.assertTrue(all(x.keys.pinned and x.values.pinned for x in cpu.layers))
        restored = adapter.to_device(cpu)
        self.assertTrue(all(x.keys.device == "gpu" for x in restored.layers))

        active = assemble_pages((restored, second), cat=fake_cat)
        self.assertEqual(active.logical_positions, (100, 101, 104, 105))
        target_recurrent = self.cache.layers[0].recurrent_states
        returned = write_active_cache(self.cache, active)
        self.assertEqual(returned, active.logical_positions)
        self.assertEqual(self.cache.layers[1].keys.data[0][0], [10, 11, 14, 15])
        self.assertIs(self.cache.layers[0].recurrent_states, target_recurrent)

    def test_extracted_page_does_not_alias_cache(self):
        page = extract_page(self.cache, 0, 2)
        page.layers[0].keys.data[0][0][0] = 999
        self.assertEqual(self.cache.layers[1].keys.data[0][0][0], 10)

    def test_extract_suffix_page_translates_sparse_physical_offset(self):
        # Physical positions 0..2 represent selected logical history; the new
        # suffix begins at logical position 100 and occupies physical 3..5.
        page = extract_suffix_page(
            self.cache,
            101,
            103,
            suffix_logical_start=100,
            physical_prefix_length=3,
        )
        self.assertEqual(page.logical_positions, (101, 102))
        self.assertEqual(page.layers[0].keys.data[0][0], [14, 15])
        with self.assertRaisesRegex(ValueError, "before the contiguous suffix"):
            extract_suffix_page(
                self.cache,
                99,
                101,
                suffix_logical_start=100,
                physical_prefix_length=3,
            )

    def test_page_order_and_missing_layers_are_rejected(self):
        early = extract_page(self.cache, 0, 2)
        late = extract_page(self.cache, 4, 6)
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            assemble_pages((late, early), cat=fake_cat)
        missing = Cache(self.cache.layers[:3])
        active = assemble_pages((early,), cat=fake_cat)
        with self.assertRaisesRegex(RuntimeError, "missing full-attention layer 3"):
            write_active_cache(missing, active)

    def test_empty_or_inconsistent_full_attention_cache_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no initialized"):
            full_attention_layers(Cache([RecurrentLayer({})]))
        with self.assertRaisesRegex(RuntimeError, "only one"):
            full_attention_layers(Cache([FullLayer(tensor(range(2)), None)]))
        self.cache.layers[3].values = tensor(range(5))
        with self.assertRaisesRegex(RuntimeError, "lengths differ"):
            extract_page(self.cache, 0, 2)

    def test_dtype_and_device_mismatch_are_rejected(self):
        self.cache.layers[1].values = tensor(range(6), dtype="fp32")
        with self.assertRaisesRegex(RuntimeError, "dtypes differ"):
            extract_page(self.cache, 0, 2)


if __name__ == "__main__":
    unittest.main()
