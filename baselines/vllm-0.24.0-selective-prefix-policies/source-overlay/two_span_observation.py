"""Development extension: semantic observation span differs from recompute span.

Unmodified shared coordinator and authenticated capture/reset/receipt channel.
The baseline internal state key 'start' is exclusively the recompute boundary B;
'semantic_start' is exclusively the actual renderer tool/template suffix start S.
No prompt tokens or logical page IDs are changed by this adapter.
"""
from dataclasses import dataclass

from contracts import (ObservationRuntime, ObservationTransition, PreparedObservation,
    _hash, _history, _turn, _vector)


@dataclass(frozen=True)
class TwoSpanTransition(ObservationTransition):
    recompute_start: int
    span_schema: int = 2


class TwoSpanObservationRuntime(ObservationRuntime):
    """One frozen-history transition; no growing-history multi-turn support."""

    def _event(self, name, binding, **fields):
        state = self._sessions.get((binding.session_id, binding.generation))
        if state is not None and "semantic_start" in state:
            # Parent recompute checker emits start=B. Publish actual semantic S
            # under observation_start and preserve B explicitly in every event.
            fields.update(observation_start=state["semantic_start"],
                          observation_end=state["end"], recompute_start=state["start"],
                          span_schema=2)
        return super()._event(name, binding, **fields)

    def prepare(self, binding, metadata):
        with self._lock:
            state = self._state(binding)
            if binding.external_hit_tokens != binding.num_pages * binding.page_tokens:
                raise ValueError("restore must hit exactly the frozen history")
            span_keys = {"observation_start", "observation_end", "recompute_start"}
            if binding.stage == "observation":
                if state["phase"] != "captured" or _turn(binding) <= state["capture_turn"]:
                    raise ValueError("observation reservation replay/order mismatch")
                if set(metadata) != span_keys | {"prior_ranked_pages", "prompt_tokens"}:
                    raise ValueError("two-span observation metadata schema mismatch")
                ranking = tuple(metadata["prior_ranked_pages"])
                if any(type(p) is not int for p in ranking) or sorted(ranking) != list(range(binding.num_pages)):
                    raise ValueError("A_t requires complete dense ranking")
                semantic, end, boundary = (metadata["observation_start"],
                    metadata["observation_end"], metadata["recompute_start"])
                tokens = metadata["prompt_tokens"]
                if (any(type(x) is not int for x in (semantic, end, boundary))
                        or not 0 <= boundary <= semantic < end
                        or boundary != semantic // binding.page_tokens * binding.page_tokens
                        or boundary != binding.external_hit_tokens
                        or type(tokens) is not list or len(tokens) != end
                        or any(type(t) is not int or t < 0 for t in tokens)
                        or _hash(tokens) != binding.canonical_prompt_hash):
                    raise ValueError("two-span observation geometry or actual prompt mismatch")
                query_hash = _hash(dict(prior=ranking, history=_history(binding),
                    observation_start=semantic, observation_end=end, recompute_start=boundary,
                    prompt=binding.canonical_prompt_hash, span_schema=2))
            elif binding.stage == "generation":
                if state["phase"] != "ranked" or _turn(binding) != state["transition"].turn:
                    raise ValueError("generation requires sealed current transition")
                if set(metadata) != span_keys | {"prompt_tokens", "transition_hash"}:
                    raise ValueError("two-span generation metadata schema mismatch")
                transition = state["transition"]
                tokens = metadata["prompt_tokens"]
                if (metadata["transition_hash"] != transition.transition_hash
                        or any(type(metadata[k]) is not int for k in span_keys)
                        or metadata["observation_start"] != transition.observation_start
                        or metadata["recompute_start"] != transition.recompute_start
                        or metadata["observation_end"] != transition.observation_end
                        or type(tokens) is not list
                        or any(type(t) is not int or t < 0 for t in tokens)
                        or _hash(tokens) != binding.canonical_prompt_hash
                        or tokens[:state["end"]] != state["observation_tokens"]):
                    raise ValueError("generation changed sealed spans or observation prefix")
                ranking, query_hash = transition.ranked_pages, transition.query_hash
            else:
                raise ValueError("unsupported Observation stage")
            return PreparedObservation(binding.reservation_id, binding.stage,
                _hash(_history(binding)), _hash(metadata), ranking, self.selected_count, query_hash)

    def consume(self, prepared, binding, metadata):
        with self._lock:
            if prepared != self.prepare(binding, metadata):
                raise ValueError("prepared two-span Observation state changed")
            state = self._state(binding)
            if binding.stage == "observation":
                state.update(phase="observing", observation_binding=binding,
                    prior_pages=tuple(sorted(prepared.ranked_pages[:self.selected_count])),
                    start=metadata["recompute_start"], semantic_start=metadata["observation_start"],
                    end=metadata["observation_end"], observation_tokens=list(metadata["prompt_tokens"]))
            else:
                state.update(phase="generating", generation_binding=binding)
            self._event("OBSERVATION_CONTROLS_CONSUMED", binding,
                        selected_pages=list(sorted(prepared.ranked_pages[:self.selected_count])))
            return dict(selective_page_contract_version=2, selective_page_policy=self.policy,
                selective_page_num_pages=binding.num_pages, selective_page_page_tokens=binding.page_tokens,
                selective_page_selected_count=self.selected_count,
                selective_page_prefix_identity=binding.native_prefix_identity,
                selective_page_ranked_pages=list(prepared.ranked_pages),
                selective_page_query_hash=prepared.query_hash)

    def accept_publication(self, publication):
        binding = publication.binding
        if binding.stage == "capture":
            return super().accept_publication(publication)
        with self._lock:
            state = self._state(binding)
            if (publication.capture_point != self.capture_point
                    or publication.provenance != self.provenance_hash):
                raise ValueError("worker capture point/provenance mismatch")
            if binding.stage != "observation" or state["phase"] != "observing":
                raise ValueError("query publication requires consumed A_t observation")
            if state["observation_binding"] != binding or tuple(publication.rows) != (state["end"] - 1,):
                raise ValueError("query must be exact final observation-token row")
            raw = publication.payload
            if (type(raw) is not dict or set(raw) != {"vectors"}
                    or not isinstance(raw["vectors"], (list, tuple)) or len(raw["vectors"]) != 1):
                raise ValueError("worker payload requires exactly one final-row vector")
            query = _vector(raw["vectors"][0], self.dimension)
            ranking = tuple(sorted(range(binding.num_pages),
                key=lambda i: (-sum(a * b for a, b in zip(query, state["pages"][i])), i)))
            selected = tuple(sorted(ranking[:self.selected_count]))
            decision = None
            bridge = getattr(self, "_remote_ra_selector_bridge", None)
            if bridge is not None:
                decision = bridge.select_observation(publication, self)
                ranking = decision.ranked_logical_pages
                selected = decision.selected_logical_pages
            transition = TwoSpanTransition(
                session_id=binding.session_id, generation=binding.generation, turn=_turn(binding),
                observation_reservation_id=binding.reservation_id, observation_request_id=binding.request_id,
                observation_prompt_hash=binding.canonical_prompt_hash,
                observation_start=state["semantic_start"], observation_end=state["end"],
                recompute_start=state["start"], previous_pages=state["prior_pages"],
                ranked_pages=ranking, selected_pages=selected,
                query_hash=(decision.query_hash if decision is not None else _hash(dict(
                    vector=raw["vectors"][0], reservation=binding.reservation_id,
                    prompt=binding.canonical_prompt_hash, row=state["end"] - 1,
                    observation_start=state["semantic_start"], recompute_start=state["start"],
                    observation_end=state["end"], provenance=self.provenance_hash, span_schema=2))),
                provenance_hash=self.provenance_hash, history_hash=_hash(_history(binding)))
            state.update(phase="ranked", transition=transition)
            self._event("OBSERVATION_TRANSITION_SEALED", binding,
                transition_hash=transition.transition_hash, prior_pages=list(state["prior_pages"]),
                selected_pages=list(selected), ranked_pages=list(ranking),
                selected_count=self.selected_count, query_hash=transition.query_hash,
                final_observation_row=state["end"] - 1)
            return transition

    # Parent record_recompute checks the actual external hit and forward coverage
    # against state['start']=B, not S. Parent finalize_success/complete require that
    # receipt and preserve the TwoSpanTransition (whose hash includes both spans).
