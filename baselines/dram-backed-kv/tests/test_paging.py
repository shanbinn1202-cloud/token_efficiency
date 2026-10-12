from __future__ import annotations

import unittest

from dram_kv.paging import TokenSegment, active_set_delta, build_page_catalog


class PagingTests(unittest.TestCase):
    def test_catalog_splits_segments_and_preserves_positions(self) -> None:
        catalog = build_page_catalog(
            "s",
            (
                TokenSegment(0, "system", 0, 10, mandatory=True),
                TokenSegment(1, "assistant", 10, 25),
            ),
            page_size=8,
            bytes_per_token=4,
        )
        self.assertEqual(
            [(p.token_start, p.token_end) for p in catalog.pages],
            [(0, 8), (8, 10), (10, 16), (16, 24), (24, 25)],
        )
        self.assertEqual(sum(p.byte_size for p in catalog.pages), 100)
        self.assertTrue(all(p.mandatory for p in catalog.pages[:2]))

    def test_delta_distinguishes_reactivation(self) -> None:
        delta = active_set_delta(("p0", "p2", "p3"), ("p0", "p1"), ("p0", "p1", "p2"))
        self.assertEqual(delta.added, ("p2", "p3"))
        self.assertEqual(delta.retained, ("p0",))
        self.assertEqual(delta.removed, ("p1",))
        self.assertEqual(delta.reactivated, ("p2",))


if __name__ == "__main__":
    unittest.main()

