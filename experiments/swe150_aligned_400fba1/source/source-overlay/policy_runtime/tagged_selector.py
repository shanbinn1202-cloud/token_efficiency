"""Engine-owned tagged selection seam; production activation remains refused.

Staging is a trusted engine operation, never an HTTP API. The live factory that
joins maintained Goal/Page metadata with authenticated worker publications is
not yet proven. CPU tests exercise this same emitted seam with explicit inputs.
"""
from dataclasses import dataclass, replace
from copy import deepcopy
import threading

from dram_kv.page_vectors import CompactVector, PageVectorIndex, PageVectorPublisher

from private_authority.protocol import Channel, EngineFence
from policy_runtime.xuanxue_selector_runtime import (
    ActiveFourPolicyRouter, QueryBinding, SelectorDecision, XuanxueSelectorRuntime,
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


@dataclass(frozen=True)
class _Staged:
    binding: object
    epoch: str
    kwargs: dict


class TaggedSelectionBridge:
    """One-use scheduler binding -> private F1 fence -> actual Selector V2."""

    def __init__(self, *, channel, fence, router, producer_provenance, input_factory):
        require(type(channel) is Channel and channel.role == "engine",
                "engine-owned private Channel required")
        require(type(fence) is EngineFence, "private EngineFence required")
        require(type(router) is ActiveFourPolicyRouter
                and type(router.query_runtime) is XuanxueSelectorRuntime,
                "exact approved four-policy selector required")
        self.channel, self.fence, self.router = channel, fence, router
        self.producer_provenance = producer_provenance
        require(callable(input_factory), "trusted engine selector input factory required")
        self.input_factory = input_factory
        self.epoch = channel.identity.epoch
        self.owner = threading.get_ident()
        self._staged, self._consumed = {}, set()
        self._pages, self._queries, self._observation_decisions = {}, {}, {}
        self.events = []

    def require_admission(self):
        require(threading.get_ident() == self.owner, "selector owner thread changed")
        self.channel._check()
        require(self.channel.identity.epoch == self.epoch
                and self.channel.phase == "MONITOR", "private epoch/phase changed")
        self.fence.require_add("registered")
        return True

    def stage(self, binding, *, selector_kwargs):
        """Trusted scheduler only; bind all selector inputs to one reservation."""
        self.require_admission()
        require(binding.policy in ("goal_query", "observation_query"),
                "query policy binding required")
        key = binding.reservation_id
        require(key not in self._staged and key not in self._consumed,
                "selector reservation cannot be reused")
        require(len(self._staged) + len(self._consumed) < 1024,
                "selector lifetime capacity reached; restart authority")
        args = deepcopy(dict(selector_kwargs))
        require("fine_index" not in args, "caller cannot supply worker vector index")
        query = args.get("binding")
        require(type(query) is QueryBinding, "exact selector QueryBinding required")
        require((query.session_id, query.generation, query.turn_id, query.policy)
                == (binding.session_id, binding.generation, binding.turn, binding.policy),
                "selector session/generation/turn differs from scheduler")
        if binding.policy == "goal_query":
            require(query.goal_hash == binding.metadata.get("goal_hash"),
                    "maintained Goal differs from authenticated probe lineage")
        require(args.get("prefix_identity") == binding.native_prefix_identity
                and args.get("expected_vllm_pages") == binding.num_pages
                and args.get("vllm_page_tokens") == binding.page_tokens,
                "selector prefix/page universe differs from scheduler")
        require(binding.external_hit_tokens == binding.num_pages * binding.page_tokens
                and binding.local_hit_tokens == 0, "selector needs actual external restore")
        require(binding.producer_provenance == self.producer_provenance.provenance_hash,
                "scheduler producer provenance differs")
        fine_provenance = args.get("provenance")
        require(all(getattr(fine_provenance, key, None) == getattr(self.producer_provenance, key)
                    for key in ("model", "revision", "hidden_layer", "projection_hash",
                                "dimension", "pooling", "dtype", "vector_source")),
                "fine/coarse vector provenance differs")
        self._staged[key] = _Staged(binding, self.epoch, args)
        if binding.session_id not in self.router.query_runtime._sessions:
            self.router.query_runtime.open_session(session_id=binding.session_id,
                policy=binding.policy, generation=binding.generation, first_turn=binding.turn)

    def stage_registered(self, binding, context):
        self.require_admission()
        if (binding.policy, binding.stage) not in {
                ("goal_query", "restore"), ("observation_query", "observation")}:
            return
        require(context["generation"] == binding.generation
                and context["nonce"] == binding.nonce
                and context["prompt_hash"] == binding.canonical_prompt_hash,
                "selector registration changed trusted renderer context")
        self.stage(binding, selector_kwargs=self.input_factory(binding, context))

    def observe_publication(self, publication, adapter):
        """Only emitted coordinator's authenticated publication branch calls here."""
        self.require_admission()
        binding = publication.binding
        require(publication.provenance == self.producer_provenance.provenance_hash,
                "worker publication provenance differs")
        prefix = (binding.session_id, binding.generation)
        vectors = tuple(CompactVector(tuple(row["values"]), "int8", row["scale"])
                        for row in publication.payload["vectors"])
        if binding.stage == "capture":
            for row, vector in zip(publication.rows, vectors, strict=True):
                require((row + 1) % binding.page_tokens == 0, "nonboundary worker page")
                page = (row + 1) // binding.page_tokens - 1
                key = prefix + (binding.cumulative_prefix_identities[page],)
                require(key not in self._pages or self._pages[key] == vector,
                        "conflicting authenticated page")
                self._pages[key] = vector
        elif binding.policy == "goal_query" and binding.stage == "query":
            require(len(vectors) == 1, "one authenticated Goal query required")
            key = prefix + (publication.metadata["goal_hash"], publication.metadata["query_payload_hash"])
            require(key not in self._queries or self._queries[key] == vectors[0],
                    "conflicting authenticated Goal query")
            self._queries[key] = vectors[0]
        elif binding.policy == "observation_query" and binding.stage == "observation":
            require(len(vectors) == 1, "one authenticated observation query required")
            state = adapter._sessions[prefix]
            transition = state["transition"]
            require(transition.observation_reservation_id == binding.reservation_id,
                    "observation publication lacks same-reservation transition")
            decision = self._observation_decisions[prefix]
            require(transition.ranked_pages == decision.ranked_logical_pages
                    and transition.selected_pages == decision.selected_logical_pages
                    and transition.query_hash == decision.query_hash,
                    "emitted Observation seal differs from actual selector")

    def select_observation(self, publication, adapter):
        """Adapter invokes after row authentication, before sealing or emitting."""
        self.require_admission()
        raw = publication.payload["vectors"]
        require(len(raw) == 1 and publication.provenance == self.producer_provenance.provenance_hash,
                "one authenticated observation vector required")
        vector = CompactVector(tuple(raw[0]["values"]), "int8", raw[0]["scale"])
        decision = self._select(publication.binding, vector)
        require(len(decision.selected_logical_pages) == adapter.selected_count,
                "Observation selector budget differs from sealed transition")
        prefix = (publication.binding.session_id, publication.binding.generation)
        self._observation_decisions[prefix] = decision
        return decision

    def _select(self, binding, query_vector):
        staged = self._staged.pop(binding.reservation_id, None)
        require(staged is not None and staged.binding == binding
                and staged.epoch == self.epoch, "unstaged or changed selector reservation")
        self._consumed.add(binding.reservation_id)
        args = dict(staged.kwargs)
        index = PageVectorIndex()
        publisher = PageVectorPublisher(index, args["provenance"])
        pages = [replace(page, page_vector_ref=None) for page in args["fine_pages"]]
        by_end = {page.logical_position_end: i for i, page in enumerate(pages)}
        prefix = (binding.session_id, binding.generation)
        for coarse, identity in enumerate(binding.cumulative_prefix_identities):
            vector = self._pages[prefix + (identity,)]
            idx = by_end[(coarse + 1) * binding.page_tokens]
            ref = publisher.publish_page(binding.session_id, pages[idx].page_id, vector)
            pages[idx] = replace(pages[idx], page_vector_ref=ref)
        publisher.publish_query(binding.session_id, binding.turn, query_vector)
        query_binding = args["binding"]
        require(query_binding.query_content_hash in (None, query_vector.content_hash),
                "staged query differs from authenticated worker capture")
        args.update(fine_pages=pages, fine_index=index,
                    binding=replace(query_binding, query_content_hash=query_vector.content_hash))
        decision = self.router.execute(binding.policy, **args)
        require(type(decision) is SelectorDecision, "approved selector decision required")
        self.require_admission()
        self.events.append((binding.request_id, self.epoch, decision.query_hash))
        # Evidence comes only after the actual approved selector consumes the
        # authenticated worker vectors, never from a client ranking field.
        from remote_ra_linux.native_authority import event
        event('SELECTOR_PLAN', request_id=binding.request_id, epoch=self.epoch,
              session_id=binding.session_id, generation=binding.generation,
              policy=binding.policy, query_hash=decision.query_hash,
              selected_logical_pages=list(decision.selected_logical_pages),
              ranked_logical_pages=list(decision.ranked_logical_pages))
        return decision

    def consume(self, binding, controls):
        self.require_admission()
        # Existing adapter consumption authenticates capture and lifecycle. It
        # does not decide which selector executes; that happens here.
        for key, expected in (
            ("selective_page_policy", binding.policy),
            ("selective_page_num_pages", binding.num_pages),
            ("selective_page_page_tokens", binding.page_tokens),
            ("selective_page_prefix_identity", binding.native_prefix_identity),
        ):
            require(controls.get(key) == expected, "authenticated controls changed " + key)
        if binding.policy == "observation_query" and binding.stage == "observation":
            require(binding.reservation_id in self._staged,
                    "current Observation selector inputs missing")
            return controls  # A_t first; current query arrives in the existing forward.
        if binding.policy == "goal_query":
            key = (binding.session_id, binding.generation,
                   binding.metadata["goal_hash"], binding.metadata["query_payload_hash"])
            decision = self._select(binding, self._queries[key])
        else:
            decision = self._observation_decisions[(binding.session_id, binding.generation)]
        selected_count = controls["selective_page_selected_count"]
        require(len(decision.selected_logical_pages) == selected_count,
                "selector budget differs from authenticated producer")
        if binding.policy == "observation_query":
            require(set(decision.selected_logical_pages) ==
                    set(controls["selective_page_ranked_pages"][:selected_count]),
                    "Observation transition mask differs from actual selector")
            require(controls["selective_page_query_hash"] == decision.query_hash,
                    "Observation generation has stale selector decision")
        self.require_admission()
        return dict(decision.kv_transfer_params)

    def cancel(self, binding):
        self._staged.pop(binding.reservation_id, None)
        self._consumed.add(binding.reservation_id)

    def release_session(self, session_id, generation):
        prefix = (session_id, generation)
        for key, value in tuple(self._staged.items()):
            if (value.binding.session_id, value.binding.generation) == prefix:
                self.cancel(value.binding)
        self._pages = {k: v for k, v in self._pages.items() if k[:2] != prefix}
        self._queries = {k: v for k, v in self._queries.items() if k[:2] != prefix}
        self._observation_decisions.pop(prefix, None)
        state = self.router.query_runtime._sessions.get(session_id)
        if state is not None and state["generation"] == generation and state["state"] == "open":
            self.router.query_runtime.close_session(session_id=session_id, generation=generation)


def private_tagged_controls(coordinator, binding, controls):
    """Called by emitted coordinator inside its existing atomic commit lock."""
    bridge = getattr(coordinator, "_remote_ra_selector_bridge", None)
    if bridge is None and not getattr(coordinator, "_remote_ra_selector_required", False):
        return controls  # unchanged static baseline
    require(type(bridge) is TaggedSelectionBridge, "dynamic private selector bridge missing")
    return bridge.consume(binding, controls)


def require_private_admission(coordinator):
    bridge = getattr(coordinator, "_remote_ra_selector_bridge", None)
    require(type(bridge) is TaggedSelectionBridge, "dynamic private selector bridge missing")
    return bridge.require_admission()


def private_worker_publication(coordinator, publication, adapter):
    bridge = getattr(coordinator, "_remote_ra_selector_bridge", None)
    if bridge is None and not getattr(coordinator, "_remote_ra_selector_required", False):
        return
    require(type(bridge) is TaggedSelectionBridge, "dynamic private selector bridge missing")
    bridge.observe_publication(publication, adapter)


def private_prepare_adapter(coordinator, publication, adapter):
    bridge = getattr(coordinator, "_remote_ra_selector_bridge", None)
    if bridge is None and not getattr(coordinator, "_remote_ra_selector_required", False):
        return
    require(type(bridge) is TaggedSelectionBridge, "dynamic private selector bridge missing")
    bridge.require_admission()
    if publication.binding.policy == "observation_query":
        from two_span_observation import TwoSpanObservationRuntime
        require(type(adapter) is TwoSpanObservationRuntime, "exact emitted two-span adapter required")
        old = getattr(adapter, "_remote_ra_selector_bridge", None)
        require(old is None or old is bridge, "Observation selector authority changed")
        adapter._remote_ra_selector_bridge = bridge


def private_cancel(coordinator, binding):
    bridge = getattr(coordinator, "_remote_ra_selector_bridge", None)
    if type(bridge) is TaggedSelectionBridge:
        bridge.cancel(binding)


def private_release(coordinator, session_id, generation):
    bridge = getattr(coordinator, "_remote_ra_selector_bridge", None)
    if type(bridge) is TaggedSelectionBridge:
        bridge.release_session(session_id, generation)


def private_registered_binding(coordinator, binding, context):
    bridge = getattr(coordinator, "_remote_ra_selector_bridge", None)
    require(type(bridge) is TaggedSelectionBridge, "dynamic private selector bridge missing")
    bridge.stage_registered(binding, context)
