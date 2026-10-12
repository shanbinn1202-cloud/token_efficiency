"""CoACT AgentDiet history reduction on the shared inference backend."""
from copy import deepcopy
from dataclasses import dataclass
import time

from baselines.vanilla.model import FullHistoryModel
from .coact.agentdiet import AgentDietCompression
from .coact.apply import COMPRESSION_STATS_DEFAULTS, ReductionApplication


@dataclass(frozen=True)
class AgentDietPolicy:
    delay_steps: int = 2
    window_before_steps: int = 1
    token_threshold: int = 500
    reflection_enable_thinking: bool = False


POLICIES = {"agentdiet": AgentDietPolicy()}


class BackendReduction(AgentDietCompression):
    def __init__(self, owner):
        self.owner = owner
        c = owner.backend.config
        g, b = c["generation"], c["backend"]
        p = owner.policy
        super().__init__(model=owner.config.model_name, api_endpoint=b["endpoint"],
            temperature=g["temperature"], top_p=g["top_p"], top_k=g["top_k"],
            min_p=None, presence_penalty=None, repetition_penalty=g["repetition_penalty"],
            max_tokens=g["max_tokens"], timeout=g["request_timeout"], max_retries=0,
            delay_steps=p.delay_steps, window_before_steps=p.window_before_steps,
            token_threshold=p.token_threshold, token_count_model=owner.config.model_name)

    def completion(self, **kwargs):
        started = time.monotonic()
        try:
            # Use the same tokenizer, pool, seed and sampling settings
            # as the agent. CoACT's Qwen helper disables reflector thinking:
            # otherwise a mention of </step> inside reasoning stops too early.
            result = self.owner.backend.complete(model=kwargs["model"],
                messages=kwargs["messages"], tools=None, temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"], stop=kwargs["stop"], _text_response=True,
                extra_body={"chat_template_kwargs": {
                    "enable_thinking": self.owner.policy.reflection_enable_thinking}})
        except Exception as exc:
            self.owner.requests.append({"kind": "agentdiet", "seconds": time.monotonic()-started,
                "usage": {}, "error": str(exc)})
            raise  # CoACT catches it and leaves history unchanged; error remains observable.
        self.owner.record(result, "agentdiet", time.monotonic()-started)
        self.owner.requests[-1]["response"] = result.model_dump()
        return result


class AgentDietModel(FullHistoryModel, ReductionApplication):
    def __init__(self, *, variant="agentdiet", policy=None, **kwargs):
        self.policy = policy or POLICIES[variant]
        super().__init__(variant=variant, policy=self.policy, **kwargs)
        self._compression_stats = deepcopy(COMPRESSION_STATS_DEFAULTS)
        self.strategy = None

    def reduce_history(self, messages):
        """Called after tool messages are appended, matching CoACT's hook."""
        if self.backend is None:
            raise ValueError("AgentDiet requires the shared inference backend")
        if self.strategy is None:
            self.strategy = BackendReduction(self)
        self.messages = messages
        reduction = self.strategy.reduce_message_in_history(messages)
        self._record_compression_attempt(reduction.attempt)
        self._apply_agentdiet_reduction(reduction)

    def _prepare_messages_for_api(self, messages):
        prepared = super()._prepare_messages_for_api(messages)
        for original, message in zip(messages, prepared, strict=True):
            message.pop("agent_erased", None)  # Archive only; never reinsert erased history.
            if (original.get("extra") or {}).get("agentdiet_reduced"):
                message.pop("reasoning_content", None)
                message.pop("reasoning", None)
        return prepared

    def serialize(self):
        return {**super().serialize(), "compression_stats": deepcopy(self._compression_stats)}
