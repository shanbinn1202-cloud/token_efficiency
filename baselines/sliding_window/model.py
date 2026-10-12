"""Keep a fixed prefix and the newest tokens within the input KV budget."""

from dataclasses import dataclass

from baselines.common.history import split_turns
from baselines.common.model import HistoryModel
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


@dataclass(frozen=True)
class WindowPolicy:
    """The common input budget determines the retained token suffix."""


POLICIES = {"sliding_window": WindowPolicy()}

class SlidingWindowMemory:
    def __init__(self, policy, kv_budget):
        if type(kv_budget) is not int or kv_budget <= 0:
            raise ValueError("sliding_window kv_budget must be a positive integer")
        self.policy = policy
        self.kv_budget = kv_budget
        self.goal_changes = 0
        self.fixed_prefix_messages = None

    def select_tokens(self, ids, *, prefix_tokens, header_tokens):
        """Preserve the fixed prefix and exact newest suffix, independent of turns."""
        if prefix_tokens + header_tokens > self.kv_budget:
            raise ValueError("sliding_window fixed prefix and generation header exceed input budget; "
                             "prefix cannot be truncated")
        if not 0 <= prefix_tokens <= len(ids) or header_tokens < 1:
            raise ValueError("Invalid prefix/header token counts")
        suffix = self.kv_budget - prefix_tokens
        selected = list(ids) if len(ids) <= self.kv_budget else list(ids[:prefix_tokens]) + list(ids[-suffix:])
        return selected, {"kv_budget": self.kv_budget, "budget_scope": "input",
            "selection": "recent_tokens", "prefix_tokens": prefix_tokens,
            "available_input_tokens": len(ids), "prompt_tokens": len(selected),
            "dropped_tokens": len(ids)-len(selected)}


class SlidingWindowModel(HistoryModel):
    def __init__(self, *, variant="sliding_window", policy=None, kv_budget=None, **kwargs):
        if kv_budget is None:
            from tebench.config import load
            kv_budget = load()["cache"]["kv_budget"]
        super().__init__(variant=variant,
                         memory=SlidingWindowMemory(policy or POLICIES[variant], kv_budget), **kwargs)

    def _prepare_messages_for_api(self, messages):
        # Count only after the same API normalization used by other methods.
        # Selection happens in _query, where per-request template options are available.
        return LitellmModel._prepare_messages_for_api(self, messages)

    def _query(self, messages, **kwargs):
        if self.backend is None:
            raise ValueError("sliding_window input budgeting requires a backend with the model tokenizer")
        from tebench.backends.base import normalize_messages
        options = self.config.model_kwargs | kwargs
        if self.memory.fixed_prefix_messages is None:
            import copy
            self.memory.fixed_prefix_messages = copy.deepcopy(split_turns(messages)[0])
        prefix = self.memory.fixed_prefix_messages
        template = {"enable_thinking": self.backend.config["model"]["enable_thinking"],
                    **(options.get("extra_body") or {}).get("chat_template_kwargs", {})}
        render = lambda generation: self.backend.tokenizer.apply_chat_template(
            normalize_messages(prefix), tools=[BASH_TOOL], tokenize=False,
            add_generation_prompt=generation, **template)
        prefix_text, first_prompt = render(False), render(True)
        text, ids = self.backend.render(messages, [BASH_TOOL], options)
        if not first_prompt.startswith(prefix_text) or not text.startswith(prefix_text):
            raise ValueError("Chat template does not preserve the fixed prefix")
        encode = lambda t: self.backend.tokenizer.encode(t, add_special_tokens=False)
        selected, trace = self.memory.select_tokens(ids, prefix_tokens=len(encode(prefix_text)),
            header_tokens=len(encode(first_prompt[len(prefix_text):])))
        self.selections.append(trace)
        return super()._query(messages, _prompt_token_ids=selected, **kwargs)



class KVSlidingWindowModel(SlidingWindowModel):
    """Keep message history intact; the persistent server selects cached KV."""

    def _query(self, messages, **kwargs):
        result = HistoryModel._query(self, messages, **kwargs)
        measurement = self.backend.last_measurement
        self.selections.append({key: measurement[key] for key in (
            "prefix_tokens", "retained_kv_tokens", "prompt_tokens", "cached_tokens",
            "prefill_tokens", "total_evicted_tokens", "logical_computed_tokens")})
        return result
