"""Actual renderer owner with file identity and tokenizer semantic checking."""
import importlib
from pathlib import Path
import sys

from dynamic_runtime import require
from runtime_renderer import EngineSourceRenderer, bind_vllm_template_callback


class ActualRendererBinding:
    def __init__(self, renderer, authority):
        self.owner, self.authority = renderer, authority
        self.tokenizer, self.model_config = renderer.tokenizer, renderer.model_config
        self.hf = importlib.import_module("vllm.renderers.hf")
        self.chat = importlib.import_module("transformers.utils.chat_template_utils")
        require(type(renderer) is self.hf.HfRenderer, "actual native HfRenderer required")
        self.methods = {name: getattr(renderer, name) for name in (
            "render_messages_async", "tokenize_prompts_async", "process_for_engine_async", "_apply_prompt_extras")}
        self.compiled = {}
        self.callback = bind_vllm_template_callback(hf_renderer_module=self.hf,
            model_config=self.model_config, tokenizer=self.tokenizer, verify_identity=self.verify)
        # Model profile (gemma-s1): a release installed for Gemma 4 names it in its
        # manifest; every other release is the Qwen profile and renders as before.
        manifest = getattr(getattr(authority, "loaded", None), "manifest", None)
        from remote_ra_live_v2.model_profile import QWEN, from_manifest, is_gemma4
        self.gemma4 = is_gemma4(from_manifest(manifest) if isinstance(manifest, dict) else QWEN)
        if self.gemma4:
            from gemma4_renderer import Gemma4SourceRenderer
            self.renderer = Gemma4SourceRenderer(model_dir=authority.model_root,
                tokenizer=self.tokenizer, render_template=self.render_template,
                verify_identity=self.verify, page_tokens=authority.smoke_config["page_tokens"])
            return
        self.renderer = EngineSourceRenderer(evidence_dir=Path(__file__).with_name("model-evidence"),
            tokenizer=self.tokenizer, render_template=self.render_template,
            verify_identity=self.verify, page_tokens=authority.smoke_config["page_tokens"])

    def verify(self):
        require(self.authority.verify_identity() is True, "live installation changed")
        require(type(self.owner) is self.hf.HfRenderer and self.owner.tokenizer is self.tokenizer
                and self.owner.model_config is self.model_config
                and all(getattr(self.owner, name) == method for name, method in self.methods.items()),
                "actual renderer owner/method changed")
        # File-level deployed module checks avoid the prior false-positive
        # transitive bytecode/namespace classifier. This is trusted-process
        # drift detection, not isolation against arbitrary Python mutation.
        names = ("vllm.renderers.hf", "vllm.renderers.base", "vllm.renderers.params",
                 "vllm.entrypoints.chat_utils", "runtime_renderer", "tokenizer_semantics",
                 "transformers.utils.chat_template_utils")
        if getattr(self, "gemma4", False):
            names += ("gemma4_renderer",)
        for name in names:
            self.authority.loaded.verify_module(name)
        for name, module in tuple(sys.modules.items()):
            if name.startswith("jinja2") and getattr(module, "__file__", "").endswith(".py"):
                self.authority.loaded.verify_module(name)
        for text, (compiled, record) in self.compiled.items():
            require(self.chat._compile_jinja_template(text) is compiled
                    and self._record(compiled) == record, "compiled template environment changed")
        return True

    @staticmethod
    def _record(compiled):
        env = compiled.environment
        return (compiled.root_render_func, compiled.root_render_func.__code__,
                tuple(sorted(env.filters.items())), tuple(sorted(env.tests.items())),
                tuple(sorted(env.globals.items())), tuple(sorted(env.extensions.items())),
                env.autoescape, env.is_async, env.optimized, env.trim_blocks, env.lstrip_blocks,
                env.undefined, env.finalize, env.loader)

    def render_template(self, template, conversation, kwargs):
        self.verify()
        compiled = self.chat._compile_jinja_template(template)
        if template not in self.compiled:
            require(len(self.compiled) < 1024, "compiled-template lifetime capacity reached")
            self.compiled[template] = (compiled, self._record(compiled))
        result = self.callback(template, conversation, kwargs)
        self.verify()
        return result
