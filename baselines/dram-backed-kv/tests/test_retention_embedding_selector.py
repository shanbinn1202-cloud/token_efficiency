from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import unittest

from dram_kv.backends import ShadowBackend
from dram_kv.schema import GoalRecord, PageCatalog, PageRecord, RetrievalGoal, TurnContext
from dram_kv.selectors import (
    EmbeddingProvenance,
    EmbeddingSelector,
    HybridSelector,
    LexicalMetadataSelector,
    RetentionConfig,
    RetentionSelector,
)
from dram_kv.selectors.embedding import DOCUMENT_SCHEMA_V2, goal_query_document, page_document


def _page(page_id: str, turn: int, start: int, word: str) -> PageRecord:
    return PageRecord(
        page_id, turn, "assistant", start, start + 4, start, start + 4, 16,
        summary=word,
    )


def _goal(turn: int, word: str) -> GoalRecord:
    retrieval = RetrievalGoal(description=word)
    return GoalRecord(
        "s", turn, "test", {}, word, root_goal="task", retrieval_goal=retrieval,
    )


def _context(turn: int, ids: tuple[str, ...]) -> TurnContext:
    return TurnContext(
        "s", "i", turn, "task", None, "inspect", "result",
        historical_turn_ids=tuple(range(1, turn)), historical_page_ids=ids,
    )


class KeywordEmbedder:
    def encode(self, text: str) -> tuple[float, float]:
        lowered = text.casefold()
        return (1.0, 0.0) if "alpha" in lowered else (0.0, 1.0)


class RetentionAndEmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = PageCatalog("s", (
            _page("alpha", 1, 0, "alpha"),
            _page("beta", 2, 4, "beta"),
        ))
        self.ids = ("alpha", "beta")

    def test_grace_delays_active_replacement_without_exceeding_budget(self) -> None:
        selector = RetentionSelector(
            LexicalMetadataSelector(page_budget=1, recent_turns=0, recent_quota=0),
            RetentionConfig(grace_turns=1),
        )
        first = selector.select(_context(3, self.ids), _goal(3, "alpha"), self.catalog)
        second = selector.select(
            _context(4, self.ids), _goal(4, "beta"), self.catalog,
            first.ordered_page_ids, first.ordered_page_ids,
        )
        third = selector.select(
            _context(5, self.ids), _goal(5, "beta"), self.catalog,
            second.ordered_page_ids, first.ordered_page_ids,
        )
        self.assertEqual(first.ordered_page_ids, ("alpha",))
        self.assertEqual(second.ordered_page_ids, ("alpha",))
        self.assertEqual(second.reasons["alpha"].source, "retention_hysteresis")
        self.assertEqual(third.ordered_page_ids, ("beta",))
        self.assertLessEqual(len(second.ordered_page_ids), 1)

    def test_standby_slack_avoids_simulated_reactivation_transfer(self) -> None:
        selector = RetentionSelector(
            LexicalMetadataSelector(page_budget=1, recent_turns=0, recent_quota=0),
            RetentionConfig(slack_pages=1),
        )
        backend = ShadowBackend()
        first = selector.select(_context(3, self.ids), _goal(3, "alpha"), self.catalog)
        backend.execute(_context(3, self.ids), self.catalog, first)
        second = selector.select(
            _context(4, self.ids), _goal(4, "beta"), self.catalog,
            first.ordered_page_ids, first.ordered_page_ids,
        )
        backend.execute(_context(4, self.ids), self.catalog, second)
        third = selector.select(
            _context(5, self.ids), _goal(5, "alpha"), self.catalog,
            second.ordered_page_ids, self.ids,
        )
        metrics = backend.execute(_context(5, self.ids), self.catalog, third)
        self.assertEqual(second.standby_page_ids, ("alpha",))
        self.assertEqual(metrics["simulated_h2d_bytes"], 0)
        self.assertEqual(metrics["resident_reactivated_pages"], 0)
        self.assertEqual(len(third.ordered_page_ids), 1)

    def test_standby_slack_expires_instead_of_pin_forever(self) -> None:
        selector = RetentionSelector(
            LexicalMetadataSelector(page_budget=1, recent_turns=0, recent_quota=0),
            RetentionConfig(slack_pages=1),
        )
        first = selector.select(_context(3, self.ids), _goal(3, "alpha"), self.catalog)
        second = selector.select(
            _context(4, self.ids), _goal(4, "beta"), self.catalog,
            first.ordered_page_ids, first.ordered_page_ids,
        )
        third = selector.select(
            _context(5, self.ids), _goal(5, "beta"), self.catalog,
            second.ordered_page_ids, self.ids,
        )
        self.assertEqual(second.standby_page_ids, ("alpha",))
        self.assertEqual(third.standby_page_ids, ())

    def test_embedding_ranks_by_cosine_and_caches_page_and_keep_query(self) -> None:
        selector = EmbeddingSelector(
            KeywordEmbedder(),
            EmbeddingProvenance("fake-semantic", "test-revision", "mean"),
            page_budget=1,
            recent_turns=0,
            recent_quota=0,
        )
        goal = _goal(3, "alpha")
        first = selector.select(_context(3, self.ids), goal, self.catalog)
        same_retrieval = replace(goal, turn_id=4, gate_decision="KEEP")
        second = selector.select(
            _context(4, self.ids), same_retrieval, self.catalog,
            first.ordered_page_ids, first.ordered_page_ids,
        )
        self.assertEqual(first.retrieved_page_ids, ("alpha",))
        self.assertEqual(second.retrieved_page_ids, ("alpha",))
        self.assertEqual(selector.page_encode_count, 2)
        self.assertEqual(selector.query_encode_count, 1)
        self.assertIn("embedding_model_hash", first.selector_config)

    def test_embedding_preserves_mandatory_and_recent_safety(self) -> None:
        mandatory = replace(self.catalog.pages[0], mandatory=True)
        catalog = PageCatalog("s", (mandatory, self.catalog.pages[1]))
        selector = EmbeddingSelector(
            KeywordEmbedder(),
            EmbeddingProvenance("fake", "r1", "mean"),
            page_budget=2,
            recent_turns=2,
            recent_quota=1,
        )
        plan = selector.select(_context(3, self.ids), _goal(3, "alpha"), catalog)
        self.assertEqual(plan.mandatory_page_ids, ("alpha",))
        self.assertEqual(plan.recent_page_ids, ("beta",))
        self.assertEqual(plan.ordered_page_ids, self.ids)

    def test_hybrid_rrf_combines_rankings_deterministically(self) -> None:
        selector = HybridSelector(
            KeywordEmbedder(),
            EmbeddingProvenance("fake", "r1", "mean"),
            page_budget=1,
            recent_turns=0,
            recent_quota=0,
            lexical_weight=2.0,
            embedding_weight=1.0,
            rrf_k=10,
        )
        plan = selector.select(_context(3, self.ids), _goal(3, "alpha"), self.catalog)
        repeated = selector.select(_context(3, self.ids), _goal(3, "alpha"), self.catalog)
        self.assertEqual(plan.retrieved_page_ids, ("alpha",))
        self.assertEqual(plan.ordered_page_ids, repeated.ordered_page_ids)
        self.assertEqual(plan.selector, "hybrid_rrf")
        self.assertEqual(plan.selector_config["fusion"], "weighted_reciprocal_rank")
        self.assertIn("lexical_rank=1", plan.reasons["alpha"].detail)

    def test_hybrid_rejects_invalid_fusion_parameters(self) -> None:
        with self.assertRaises(ValueError):
            HybridSelector(
                KeywordEmbedder(), EmbeddingProvenance("fake", "r1", "mean"),
                page_budget=1, lexical_weight=0,
            )

    def test_v2_embedding_documents_put_typed_fields_before_bounded_prose(self) -> None:
        page = replace(self.catalog.pages[0], files=("pkg/core.py",), summary="x" * 2_000)
        page_text = page_document(page, DOCUMENT_SCHEMA_V2)
        query_text = goal_query_document(_goal(3, "alpha"), DOCUMENT_SCHEMA_V2)
        self.assertLess(page_text.index("files:"), page_text.index("summary:"))
        self.assertLess(len(page_text), 2_000)
        self.assertTrue(query_text.startswith("query_type:"))

        selector = EmbeddingSelector(
            KeywordEmbedder(),
            EmbeddingProvenance(
                "fake", "r1", "mean", document_schema=DOCUMENT_SCHEMA_V2,
            ),
            page_budget=1,
            recent_turns=0,
            recent_quota=0,
        )
        plan = selector.select(
            _context(3, self.ids), _goal(3, "alpha"), self.catalog,
        )
        selected = self.catalog.by_id()[plan.retrieved_page_ids[0]]
        expected = sha256(
            page_document(selected, DOCUMENT_SCHEMA_V2).encode("utf-8")
        ).hexdigest()
        self.assertIn(f"document_sha256={expected}", plan.reasons[selected.page_id].detail)


if __name__ == "__main__":
    unittest.main()
