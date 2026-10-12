from __future__ import annotations

from dataclasses import asdict, replace
import json
import unittest

from dram_kv.backends import ShadowBackend
from dram_kv.page_vectors import (
    CompactVector,
    PageVectorIndex,
    PageVectorProvenance,
    PageVectorPublisher,
    SimulatedPrefillPageVectorPublisher,
)
from dram_kv.pipeline import ExperimentPipeline
from dram_kv.schema import (
    GoalRecord,
    PageCatalog,
    PageRecord,
    RetrievalGoal,
    TurnContext,
)
from dram_kv.selectors import (
    PrefillPageEmbeddingSelectorV2,
    PrefillPageHybridSelectorV2,
)


TRACE_HASH = "a" * 64


def page(
    page_id: str,
    turn_id: int,
    start: int,
    *,
    mandatory: bool = False,
    files: tuple[str, ...] = (),
    summary: str | None = None,
    page_vector_ref: str | None = None,
) -> PageRecord:
    return PageRecord(
        page_id=page_id,
        turn_id=turn_id,
        role="tool",
        token_start=start,
        token_end=start + 16,
        logical_position_start=start,
        logical_position_end=start + 16,
        byte_size=64,
        mandatory=mandatory,
        files=files,
        summary=summary,
        page_vector_ref=page_vector_ref,
    )


def context(catalog: PageCatalog, turn_id: int = 4) -> TurnContext:
    history = catalog.historical_for_turn(turn_id)
    return TurnContext(
        session_id=catalog.session_id,
        instance_id="instance",
        turn_id=turn_id,
        task_objective="fix cache",
        previous_goal_text="find cache evidence",
        previous_action="inspect",
        current_observation="latest tool observation",
        historical_turn_ids=tuple(sorted({item.turn_id for item in history})),
        historical_page_ids=tuple(item.page_id for item in history),
    )


def goal(turn_id: int = 4) -> GoalRecord:
    retrieval = RetrievalGoal(
        "find cache evidence",
        files=("cache.py",),
        evidence_needed=("failing test",),
    )
    return GoalRecord(
        session_id="s",
        turn_id=turn_id,
        constructor="frozen-test",
        constructor_config={"trace_sha256": TRACE_HASH},
        text=retrieval.description,
        root_goal="fix cache",
        retrieval_goal=retrieval,
        gate_decision="KEEP",
        applied_decision="KEEP",
        reused_previous=True,
    )


def identity_provenance(*, revision: str = "r1") -> PageVectorProvenance:
    return PageVectorProvenance(
        model="main-model",
        revision=revision,
        hidden_layer=12,
        pooling="masked_mean",
        projection_hash="b" * 64,
        dimension=2,
        dtype="int8",
    )


class FrozenConstructor:
    def __init__(self, record: GoalRecord) -> None:
        self.record = record

    def construct(self, turn_context: TurnContext) -> GoalRecord:
        if (turn_context.session_id, turn_context.turn_id) != (
            self.record.session_id,
            self.record.turn_id,
        ):
            raise ValueError("wrong frozen Goal boundary")
        return self.record


class PageVectorPublisherTests(unittest.TestCase):
    def test_simulated_publisher_accumulates_chunks_and_accepted_decode_only(self) -> None:
        index = PageVectorIndex()
        simulated = SimulatedPrefillPageVectorPublisher(
            index,
            model="main-model",
            revision="r1",
            hidden_layer=12,
            projection=((1.0, 0.0), (0.0, 1.0)),
        )
        first = [[1.0, 0.0]] * 8
        speculative = [[1.0, 0.0], [99.0, 99.0], [1.0, 0.0]]
        self.assertEqual(simulated.observe_page_chunk("s", "p0", first), 8)
        self.assertEqual(
            simulated.observe_page_chunk(
                "s", "p0", speculative, accepted=(True, False, True)
            ),
            10,
        )
        self.assertEqual(
            simulated.observe_page_chunk("s", "p0", [[1.0, 0.0]] * 6),
            16,
        )
        reference = simulated.seal_page("s", "p0")
        self.assertTrue(reference.startswith("page-vector-v2:"))

        simulated.observe_query_chunk("s", 4, [[1.0, 0.0]])
        simulated.observe_query_chunk("s", 4, [[1.0, 0.0]])
        simulated.seal_query("s", 4)
        snapshot = index.snapshot("s", 4, ("p0",), simulated.provenance)
        self.assertTrue(snapshot.complete)
        self.assertEqual(snapshot.pages["p0"].vector.values, (127, 0))

    def test_index_is_session_scoped_and_sealed_vectors_are_immutable(self) -> None:
        index = PageVectorIndex()
        publisher = PageVectorPublisher(index, identity_provenance())
        publisher.publish_page("s", "p0", CompactVector((127, 0), "int8", 1 / 127))
        # Exact replay is idempotent, but changing a sealed Page is forbidden.
        publisher.publish_page("s", "p0", CompactVector((127, 0), "int8", 1 / 127))
        with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
            publisher.publish_page(
                "s", "p0", CompactVector((0, 127), "int8", 1 / 127)
            )
        other = index.snapshot("other", 4, ("p0",), identity_provenance())
        self.assertEqual(other.missing_page_ids, ("p0",))


class PrefillPageSelectorV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = PageVectorIndex()
        self.provenance = identity_provenance()
        self.publisher = PageVectorPublisher(self.index, self.provenance)

    def publish(self, page_id: str, values: tuple[int, int]) -> str:
        return self.publisher.publish_page(
            "s", page_id, CompactVector(values, "int8", 1 / 127)
        )

    def publish_query(self, values: tuple[int, int] = (127, 0)) -> None:
        self.publisher.publish_query(
            "s", 4, CompactVector(values, "int8", 1 / 127)
        )

    def test_embedding_v2_ranks_physical_pages_and_records_provenance(self) -> None:
        refs = {
            "mandatory": self.publish("mandatory", (0, 127)),
            "semantic": self.publish("semantic", (127, 0)),
            "other": self.publish("other", (0, 127)),
        }
        self.publish_query()
        catalog = PageCatalog("s", (
            page("mandatory", 0, 0, mandatory=True, page_vector_ref=refs["mandatory"]),
            page("semantic", 1, 16, page_vector_ref=refs["semantic"]),
            page("other", 2, 32, page_vector_ref=refs["other"]),
        ))
        selector = PrefillPageEmbeddingSelectorV2(
            self.index,
            self.provenance,
            frozen_goal_trace_hash=TRACE_HASH,
            page_budget=2,
            recent_turns=0,
            recent_quota=0,
            query_source="goal_hidden",
        )
        plan = selector.select(context(catalog), goal(), catalog)
        self.assertEqual(plan.ordered_page_ids, ("mandatory", "semantic"))
        self.assertEqual(plan.retrieved_page_ids, ("semantic",))
        self.assertEqual(plan.reasons["semantic"].source, "prefill_page_cosine_v2")
        self.assertEqual(plan.selector_config["effective_mode"], "prefill_page_cosine")
        self.assertEqual(plan.selector_config["page_size"], 16)
        self.assertEqual(plan.selector_config["hidden_layer"], 12)
        self.assertEqual(plan.selector_config["projection_hash"], "b" * 64)
        self.assertEqual(plan.selector_config["frozen_goal_trace_hash"], TRACE_HASH)
        self.assertEqual(plan.selector_config["query_source"], "goal_hidden")
        self.assertEqual(len(plan.selector_config["selector_config_hash"]), 64)

    def test_query_source_rejects_an_unknown_vector_origin(self) -> None:
        with self.assertRaisesRegex(ValueError, "query_source"):
            PrefillPageEmbeddingSelectorV2(
                self.index,
                self.provenance,
                frozen_goal_trace_hash=TRACE_HASH,
                page_budget=1,
                query_source="slm_embedding",
            )

    def test_missing_or_mismatched_vectors_fall_back_deterministically(self) -> None:
        catalog = PageCatalog("s", (
            page("first", 1, 0, files=("other.py",)),
            page("lexical", 2, 16, files=("cache.py",)),
        ))
        selector = PrefillPageEmbeddingSelectorV2(
            self.index,
            self.provenance,
            frozen_goal_trace_hash=TRACE_HASH,
            page_budget=1,
            recent_turns=0,
            recent_quota=0,
        )
        first = selector.select(context(catalog), goal(), catalog)
        second = selector.select(context(catalog), goal(), catalog)
        self.assertEqual(first.ordered_page_ids, ("lexical",))
        self.assertEqual(first.ordered_page_ids, second.ordered_page_ids)
        self.assertEqual(
            first.selector_config["effective_mode"],
            "lexical_deterministic_fallback",
        )
        self.assertEqual(first.selector_config["fallback_reason"], "query_vector_missing")

        # A query published from another model revision also fails closed.
        other = PageVectorPublisher(self.index, identity_provenance(revision="other"))
        other.publish_query("s", 4, CompactVector((127, 0), "int8", 1 / 127))
        mismatched = selector.select(context(catalog), goal(), catalog)
        self.assertEqual(
            mismatched.selector_config["fallback_reason"],
            "query_provenance_mismatch",
        )
        self.assertEqual(mismatched.ordered_page_ids, ("lexical",))

    def test_page_provenance_mismatch_falls_back_before_mixed_scoring(self) -> None:
        self.publish_query()
        other = PageVectorPublisher(self.index, identity_provenance(revision="other"))
        other.publish_page(
            "s", "lexical", CompactVector((127, 0), "int8", 1 / 127)
        )
        catalog = PageCatalog("s", (
            page("lexical", 1, 0, files=("cache.py",)),
        ))
        selector = PrefillPageEmbeddingSelectorV2(
            self.index,
            self.provenance,
            frozen_goal_trace_hash=TRACE_HASH,
            page_budget=1,
            recent_turns=0,
            recent_quota=0,
        )
        plan = selector.select(context(catalog), goal(), catalog)
        self.assertEqual(plan.ordered_page_ids, ("lexical",))
        self.assertEqual(
            plan.selector_config["fallback_reason"],
            "page_provenance_mismatch:lexical",
        )

    def test_hybrid_v2_uses_page_local_lexical_rank_and_default_three_to_one(self) -> None:
        self.publish("semantic", (127, 0))
        self.publish("lexical", (0, 127))
        self.publish_query()
        catalog = PageCatalog("s", (
            page(
                "semantic",
                1,
                0,
                files=("other.py",),
                # A V1-style duplicated summary must not enter V2 lexical rank.
                summary="cache.py cache.py cache.py",
            ),
            page("lexical", 2, 16, files=("cache.py",)),
        ))
        selector = PrefillPageHybridSelectorV2(
            self.index,
            self.provenance,
            frozen_goal_trace_hash=TRACE_HASH,
            page_budget=1,
            recent_turns=0,
            recent_quota=0,
        )
        plan = selector.select(context(catalog), goal(), catalog)
        self.assertEqual(plan.retrieved_page_ids, ("lexical",))
        self.assertEqual(plan.selector_config["lexical_weight"], 3.0)
        self.assertEqual(plan.selector_config["semantic_weight"], 1.0)
        self.assertEqual(plan.selector_config["rrf_k"], 60)

    def test_shadow_pipeline_keeps_frozen_goal_bytes_and_runtime_decoupled(self) -> None:
        self.publish("semantic", (127, 0))
        self.publish("other", (0, 127))
        self.publish_query()
        catalog = PageCatalog("s", (
            page("semantic", 1, 0),
            page("other", 2, 16),
        ))
        frozen_goal = goal()
        before = json.dumps(asdict(frozen_goal), sort_keys=True, separators=(",", ":"))
        pipeline = ExperimentPipeline(
            FrozenConstructor(frozen_goal),
            PrefillPageEmbeddingSelectorV2(
                self.index,
                self.provenance,
                frozen_goal_trace_hash=TRACE_HASH,
                page_budget=1,
                recent_turns=0,
                recent_quota=0,
            ),
            ShadowBackend({("s", 4): ("semantic",)}),
        )
        result = pipeline.run_turn(context(catalog), catalog)
        after = json.dumps(asdict(result.goal), sort_keys=True, separators=(",", ":"))
        self.assertEqual(before, after)
        self.assertEqual(result.backend_metrics["evidence_recall"], 1.0)
        self.assertTrue(result.backend_metrics["dense_context_unchanged"])
        self.assertEqual(result.selection.ordered_page_ids, ("semantic",))

    def test_page_vector_reference_mismatch_falls_back(self) -> None:
        self.publish("first", (127, 0))
        self.publish_query()
        catalog = PageCatalog("s", (
            page(
                "first",
                1,
                0,
                files=("cache.py",),
                page_vector_ref="page-vector-v2:" + "0" * 64,
            ),
        ))
        selector = PrefillPageEmbeddingSelectorV2(
            self.index,
            self.provenance,
            frozen_goal_trace_hash=TRACE_HASH,
            page_budget=1,
            recent_turns=0,
            recent_quota=0,
        )
        plan = selector.select(context(catalog), goal(), catalog)
        self.assertEqual(
            plan.selector_config["fallback_reason"],
            "page_vector_ref_mismatch:first",
        )

    def test_non_v2_physical_page_size_is_an_auditable_fallback(self) -> None:
        short_page = PageRecord("short", 1, "tool", 0, 8, 0, 8, 32, files=("cache.py",))
        catalog = PageCatalog("s", (short_page,))
        selector = PrefillPageEmbeddingSelectorV2(
            self.index,
            self.provenance,
            frozen_goal_trace_hash=TRACE_HASH,
            page_budget=1,
            recent_turns=0,
            recent_quota=0,
        )
        plan = selector.select(context(catalog), goal(), catalog)
        self.assertEqual(plan.ordered_page_ids, ("short",))
        self.assertEqual(
            plan.selector_config["fallback_reason"],
            "physical_page_size_mismatch:short",
        )


if __name__ == "__main__":
    unittest.main()
