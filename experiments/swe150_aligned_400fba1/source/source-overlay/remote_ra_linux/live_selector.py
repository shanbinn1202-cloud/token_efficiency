"""Trusted frozen-smoke source registration and Selector V2 input production.

One fresh private session runs exactly three engine requests. This does not
implement growing-history SWE-bench or authorize a production deployment.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import threading

from contracts import canonical_prompt_hash
from dynamic_runtime import DynamicContextProvider, require, source_digest
from dram_kv.page_vectors import PageVectorProvenance
from dram_kv.schema import PageRecord
from goal_adapter import capture_digest, construct_goal
from policy_runtime.xuanxue_selector_runtime import QueryBinding, observation_source_sha256
from runtime_renderer import SOURCE_SCHEMA, PLAIN_GOAL_SCHEMA, TOKEN_OPTIONS


def load_smoke_config():
    config = json.loads(Path(__file__).with_name("smoke_session.json").read_text(encoding="utf-8"))
    require(config["schema"] == "remote-ra-frozen-smoke-v1"
            and config["selected_count"] == 2 and config["projection_seed"] == 42
            and config["page_tokens"] == 784, "sealed smoke geometry differs")
    return config


def smoke_messages(config, policy):
    require(policy in ("full", "random50", "goal_query", "observation_query"), "unknown smoke policy")
    messages = [dict(role="system", content=config["system"]),
                dict(role="user", content=config["task"]["problem_statement"] + "\n" +
                     config["history_line"] * config["history_repetitions"])]
    for index in range(2 if policy == "goal_query" else 1):
        call_id = "smoke-call-" + str(index + 1)
        messages.extend([dict(role="assistant", content="", tool_calls=[dict(
            id=call_id, type="function", function=dict(name="bash", arguments={
                "command": config["commands"][index]}))]), dict(role="tool",
            tool_call_id=call_id, content=config["observations"][index])])
    return messages


def normalized_source(messages):
    # Match the actual serving ChatParams: merge_kwargs discards unset tools=None.
    # Keep omission exact rather than treating absent/None as equivalent for an
    # arbitrary template. This builder is shared by the client and engine.
    return dict(schema=SOURCE_SCHEMA, conversation=deepcopy(messages),
                template_kwargs=dict(add_generation_prompt=True, enable_thinking=False),
                tokenization_options=dict(TOKEN_OPTIONS))


def derive_goal(source, config, session_id, policy, inventory_hash):
    """Replay only the exact registered conversation, with raw status untrusted."""
    expected = normalized_source(smoke_messages(config, policy))
    require(source == expected, "registered source differs from sealed smoke chronology")
    messages = deepcopy(source["conversation"])
    for message in messages:
        for call in message.get("tool_calls", ()):
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
    replay = dict(schema_version=2, kind="frozen_agent_message_inputs", condition="full",
                  instance_id=config["task"]["instance_id"], source_inventory_sha256=inventory_hash, calls=[])
    for end in range(2, len(messages) + 1, 2):
        request = dict(model="sealed-smoke", messages=messages[:end])
        replay["calls"].append(dict(request=request, request_sha256=capture_digest(request),
            request_order_sha256=capture_digest(request, ordered=True), observation_messages=[
                dict(message_index=i, message=m) for i, m in enumerate(request["messages"]) if m["role"] == "tool"]))
    return construct_goal(replay, config["task"], len(replay["calls"]) - 1, session_id=session_id)


def smoke_request_sequence(config, policy, session_id, inventory_hash, model):
    """Pure request builder: no session, rank, span, stage or authority HTTP fields."""
    source = normalized_source(smoke_messages(config, policy))
    goal = derive_goal(source, config, session_id, policy, inventory_hash)
    messages = deepcopy(source["conversation"])
    for message in messages:
        for call in message.get("tool_calls", ()):
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
    def body(value):
        return dict(model=model, messages=deepcopy(value), stream=False, n=1, max_tokens=1,
                    temperature=0, chat_template_kwargs=dict(enable_thinking=False))
    middle = [dict(role="user", content=goal["goal_probe_text"])] if policy == "goal_query" else messages
    return (body(messages), body(middle), body(messages))


class SmokeContextProvider(DynamicContextProvider):
    """Engine derives stages/metadata from its sealed config and actual source."""
    def __init__(self, *, smoke_config, policy, session_id, epoch, **kwargs):
        super().__init__(**kwargs)
        self.smoke_config, self.policy, self.session_id = deepcopy(smoke_config), policy, session_id
        self.epoch, self.owner = epoch, threading.get_ident()
        self.history_source = normalized_source(smoke_messages(smoke_config, policy))
        self.goal = derive_goal(self.history_source, smoke_config, session_id, policy, self.renderer_identity)
        self.probe_source = dict(schema=PLAIN_GOAL_SCHEMA, text=self.goal["goal_probe_text"])
        rendered = self._render(self.history_source)
        boundary = (len(rendered.token_ids) if policy == "goal_query" else rendered.observation_start)
        require(type(boundary) is int and boundary > 0, "actual semantic history boundary required")
        self.num_pages = boundary // self.provenance.page_tokens
        require(len(rendered.token_ids) > self.num_pages * self.provenance.page_tokens,
                "smoke history needs a real suffix")
        require(self.num_pages > self.selected_count, "smoke needs a selective historical page budget")
        self.generation = None  # Assembly must start with fresh, empty authority.
        self._registered_sources = {}
        if policy == "goal_query":
            tokens = list(self._render(self.probe_source).token_ids)
            self.lineage = dict(goal_hash=self.goal["goal_hash"], query_payload_hash=sha256(json.dumps(
                dict(prompt_token_ids=tokens, add_special_tokens=False), sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def register(self, *, session_id, request_id, source, metadata=None):
        require(threading.get_ident() == self.owner and session_id == self.session_id,
                "smoke registration owner/session differs")
        require(metadata in (None, {}), "frontend cannot supply selector metadata")
        with self.coordinator.lock:
            if self.generation is None:
                from policy_runtime.tagged_selector import require_private_admission
                require_private_admission(self.coordinator)
                require(source == self.history_source,
                        "first smoke registration must be the exact capture source")
                self.generation = self.open_session(session_id=session_id, policy=self.policy,
                                                    num_pages=self.num_pages)
            old = self._registered_sources.get(request_id)
            index = old[0] if old else self._sessions[session_id]["next_stage"]
            require(index < 3, "frozen smoke cycle exhausted; new authority required")
            expected = self.probe_source if self.policy == "goal_query" and index == 1 else self.history_source
            require(source == expected, "wrong source for trusted smoke stage")
            derived = {}
            if self.policy == "goal_query" and index:
                derived = dict(self.lineage)
                if index == 2:
                    derived.update(selected_count=self.selected_count)
            elif self.policy == "observation_query" and index == 1:
                # A_t seed for the observation forward only; the final selection
                # is produced from the current authenticated hidden-state query.
                derived = dict(prior_ranked_pages=list(range(self.num_pages)))
            receipt = super().register(session_id=session_id, request_id=request_id,
                                       source=source, metadata=derived)
            self._registered_sources[request_id] = (index, source_digest(source), deepcopy(source))
            return receipt

    def _event(self, name, **fields):
        require("epoch" not in fields or fields["epoch"] == self.epoch, "provider evidence epoch changed")
        fields["epoch"] = self.epoch
        super()._event(name, **fields)
        from .native_authority import event
        event(name, **fields)


class SelectorInputFactory:
    """No callable fixtures or client Page metadata enter this engine factory."""
    def __init__(self, provider, authority):
        require(type(provider) is SmokeContextProvider, "canonical smoke provider required")
        self.provider, self.authority = provider, authority
        self.epoch = authority.identity.epoch
        values = asdict(provider.provenance)
        self.provenance = PageVectorProvenance(**{name: values[name] for name in (
            "model", "revision", "hidden_layer", "projection_hash", "dimension", "pooling", "dtype", "vector_source")})

    def __call__(self, binding, context):
        provider = self.provider
        require(self.authority.verify_admission() is True and self.authority.identity.epoch == self.epoch
                and provider.epoch == self.epoch, "selector epoch/admission changed")
        entry = provider._entries.get(binding.request_id)
        require(entry is not None and entry["state"] == "admitted"
                and provider._contexts.get(binding.request_id) == context
                and (binding.session_id, binding.generation, binding.policy) == (
                    provider.session_id, provider.generation, provider.policy)
                and canonical_prompt_hash(entry["tokens"]) == binding.canonical_prompt_hash,
                "selector lacks exact registered request/context")
        registered = provider._registered_sources[binding.request_id]
        require(registered[0] == (2 if binding.policy == "goal_query" else 1)
                and binding.stage == ("restore" if binding.policy == "goal_query" else "observation"),
                "selector request is not the registered selection stage")
        require(registered[1] == entry["receipt"]["source_sha256"]
                and source_digest(registered[2]) == registered[1],
                "selector history source differs")
        goal = derive_goal(registered[2], provider.smoke_config, provider.session_id,
                           provider.policy, provider.renderer_identity)
        require(goal["goal_hash"] == provider.goal["goal_hash"], "maintained Goal changed")
        require(binding.turn == goal["contexts"][-1]["turn_id"], "selector semantic turn differs")
        count = binding.num_pages * binding.page_tokens
        require(count % 16 == 0 and count <= len(entry["tokens"]), "fine-page geometry differs")
        tokenizer = provider._render.tokenizer
        # These records describe exact token intervals of the mixed rendered
        # history captured as runtime turn 1. Role is explicitly mixed, and
        # byte_size=0 means unknown (unused by this fixed page-count selector).
        # No role, KV byte measurement or tool outcome is inferred from text.
        pages = tuple(PageRecord(page_id=f"fine-{i // 16:08d}", turn_id=1, role="mixed",
            token_start=i, token_end=i + 16, logical_position_start=i, logical_position_end=i + 16,
            byte_size=0, summary=tokenizer.decode(entry["tokens"][i:i + 16], skip_special_tokens=False))
            for i in range(0, count, 16))
        from .native_authority import event
        event("SELECTOR_INPUT_BOUND", request_id=binding.request_id, epoch=self.epoch,
              session_id=binding.session_id, generation=binding.generation,
              turn=binding.turn, policy=binding.policy, source_sha256=registered[1],
              goal_hash=goal["goal_hash"], fine_page_count=len(pages))
        return dict(goal_output=goal, binding=QueryBinding(session_id=binding.session_id,
            generation=binding.generation, turn_id=binding.turn, policy=binding.policy,
            query_source="goal_hidden" if binding.policy == "goal_query" else "observation_hidden",
            producer_provenance_hash=self.provenance.provenance_hash,
            source_sha256=goal["goal_probe_text_sha256"] if binding.policy == "goal_query" else observation_source_sha256(goal),
            goal_hash=goal["goal_hash"], query_content_hash=None), fine_pages=pages,
            provenance=self.provenance, selected_count=provider.selected_count,
            vllm_page_tokens=binding.page_tokens, expected_vllm_pages=binding.num_pages,
            prefix_identity=binding.native_prefix_identity, recent_turns=0, recent_quota=0)
