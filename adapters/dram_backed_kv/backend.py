"""Adapt upstream Online Selector V2 to shared tools, configuration and metrics.

The upstream selector, page store, transfers and logical RoPE implementation are
imported read-only. The adapter provides token-budget and chat-format boundaries.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from tebench.config import ROOT
from tebench.backends.base import normalize_messages, response
from tebench.backends.transformers import TransformersBackend
from tebench.backends.execution import sdpa_context


def load_upstream():
    repo = ROOT / "baselines/dram-backed-kv"
    source = str(repo / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    name = "tebench_dram_upstream_online"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, repo / "scripts/serve_online_selector_v2.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class DramBackend(TransformersBackend):
    def __init__(self, config):
        super().__init__(config)
        upstream = load_upstream()
        upstream.install_qwen35_logical_rope(self.model)
        self.session = None
        self.tools = None
        self.temporary = tempfile.TemporaryDirectory(prefix="tebench-dram-")
        owner = self

        class Session(upstream.OnlineSelectorSession):
            def _render(self, messages, add_prompt):
                return upstream._extract_ids(owner.tokenizer.apply_chat_template(normalize_messages(messages), tools=owner.tools,
                    tokenize=True, add_generation_prompt=add_prompt,
                    enable_thinking=config["model"]["enable_thinking"]))

            def _plan(self, context, goal):
                # Express a fixed token budget through upstream's ratio argument.
                # Mandatory prefix constraints remain those of the original method.
                history_tokens = len(context.historical_page_ids) * upstream.PAGE_SIZE
                if config["methods"]["dram_backed_kv"]["options"]["budget_mode"] == "fixed":
                    self.args.active_ratio = min(1.0, config["cache"]["kv_budget"] / max(1, history_tokens))
                return super()._plan(context, goal)

        self.session_class = Session
        self.args = SimpleNamespace(model=config["model"]["path"], selection_mode="selector",
            switch_threshold=config["cache"]["kv_budget"], max_new_tokens=config["generation"]["max_tokens"],
            metrics=Path(self.temporary.name) / "requests.jsonl",
            **config["methods"]["dram_backed_kv"]["options"])

    def reset(self):
        if getattr(self, "session", None) is not None:
            self.session.close()
            self.session = None
        super().reset()

    def complete(self, *, model, messages, tools=None, **options):
        _, ids, max_tokens = self.prepare(messages, tools, options)
        self.tools = tools
        if self.session is not None and not self.session.accepts(messages):
            self.reset()
        if self.session is None:
            self.session = self.session_class(self.args, self.model, self.tokenizer, self.torch)
        self.args.max_new_tokens = max_tokens
        try:
            with self.torch.inference_mode(), sdpa_context(self.config["backend"]["sdpa_kernel"]):
                text, measurement = self.session.query(messages)
        except Exception:
            self.reset()
            raise
        self.torch.cuda.synchronize()
        self.last_measurement = measurement
        count = measurement["generated_tokens"]
        return response(model, text, len(ids), count, tools,
            finish_reason="length" if count == max_tokens else "stop")

    def close(self):
        self.reset()
        self.temporary.cleanup()
        super().close()
