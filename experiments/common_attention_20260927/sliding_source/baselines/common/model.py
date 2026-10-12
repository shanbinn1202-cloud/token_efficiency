"""Method-independent model accounting, history selection and pruning."""
import time
from pathlib import Path
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.models.litellm_model import LitellmModel

class HistoryModel(LitellmModel):
    abort_exceptions = [*LitellmModel.abort_exceptions, InterruptAgentFlow]

    def __init__(self, *, variant, memory=None, stop_path=None, backend=None, **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self.memory = memory
        self.stop_path = Path(stop_path) if stop_path else None
        self.requests = []
        self.selections = []
        self.backend = backend

    def completion(self, **kwargs):
        if self.backend is not None:
            return self.backend.complete(**kwargs)
        import litellm
        return litellm.completion(**kwargs)

    def gate(self, previous, latest):
        return False

    def _prepare_messages_for_api(self, messages):
        selected, trace = self.memory.select(messages, self.gate)
        self.selections.append(trace)
        return super()._prepare_messages_for_api(selected)

    def _query(self, messages, **kwargs):
        start = time.monotonic()
        try:
            if self.backend is None:
                response = super()._query(messages, **kwargs)
            else:
                from minisweagent.models.utils.actions_toolcall import BASH_TOOL
                response = self.completion(model=self.config.model_name, messages=messages,
                    tools=[BASH_TOOL], **(self.config.model_kwargs | kwargs))
        except InterruptAgentFlow:
            # Budget/turn/context termination is control flow, not a failed
            # inference request. The runner can still recover the worktree patch.
            raise
        except Exception as exc:
            self.requests.append({"kind": "agent", "seconds": time.monotonic() - start,
                                  "error": str(exc), "usage": {}})
            raise
        self.record(response, "agent", time.monotonic() - start)
        return response

    def query(self, messages, **kwargs):
        # A recorded experiment-level pruning decision stops every worker of
        # that variant at its next step, without stopping the other variants.
        if self.stop_path and self.stop_path.exists():
            raise InterruptAgentFlow(
                {
                    "role": "exit",
                    "content": "TrialPruned",
                    "extra": {"exit_status": "TrialPruned", "submission": ""},
                }
            )
        return super().query(messages, **kwargs)


    def record(self, response, kind, elapsed):
        usage = response.usage.model_dump() if response.usage else {}
        self.requests.append({"kind": kind, "seconds": elapsed, "usage": usage,
            "backend": dict(getattr(self.backend, "last_measurement", {}))})


    def serialize(self):
        return {
            **super().serialize(),
            "memory": {
                "variant": self.variant,
                "requests": self.requests,
                "selections": self.selections,
                "goal_changes": self.memory.goal_changes,
            },
        }
