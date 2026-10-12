"""vLLM completions transport: identical local chat template and tool parser."""
import time
import uuid
from .base import Backend, response


class VLLMBackend(Backend):
    @property
    def _cache_salt(self):
        if not hasattr(self._local, 'cache_salt'):
            self._local.cache_salt = uuid.uuid4().hex
        return self._local.cache_salt

    @_cache_salt.setter
    def _cache_salt(self, value):
        self._local.cache_salt = value

    def __init__(self, config):
        super().__init__(config)
        from openai import OpenAI
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"],
            local_files_only=config["model"]["local_files_only"])
        self.client = OpenAI(base_url=config["backend"]["endpoint"], api_key=config["backend"]["api_key"],
            timeout=config["generation"]["request_timeout"], max_retries=0)
        # Prefix blocks are reusable within one SWE session, but must not leak
        # between independent benchmark instances that share the system prompt.
        self._cache_salt = uuid.uuid4().hex

    def complete(self, *, model, messages, tools=None, **options):
        started = time.perf_counter()
        _, ids, max_tokens = self.prepare(messages, tools, options)
        extra = {"skip_special_tokens": False, "top_k": self.config["generation"]["top_k"],
                 "repetition_penalty": self.config["generation"]["repetition_penalty"]}
        structured = (options.get("extra_body") or {}).get("structured_outputs")
        if structured:
            extra["structured_outputs"] = structured
        result = self.client.completions.create(model=self.config["model"]["name"].split("/")[-1],
            prompt=ids, max_tokens=max_tokens, temperature=options.get("temperature", self.config["generation"]["temperature"]),
            top_p=self.config["generation"]["top_p"], seed=self.config["generation"]["seed"],
            extra_body={**extra, "cache_salt": self._cache_salt},
            **({"stop": options["stop"]} if options.get("stop") else {}))
        usage, choice = result.usage, result.choices[0]
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        self.last_measurement = {"seconds": time.perf_counter() - started,
            "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
            "cached_tokens": cached, "prefill_tokens": usage.prompt_tokens - cached}
        return response(model, choice.text, usage.prompt_tokens, usage.completion_tokens, tools,
            cached_tokens=cached, finish_reason=choice.finish_reason,
            text_only=options.get("_text_response", False),
            enable_thinking=(options.get("extra_body") or {}).get("chat_template_kwargs", {}).get(
                "enable_thinking", self.config["model"]["enable_thinking"]))

    def reset(self):
        super().reset()
        self._cache_salt = uuid.uuid4().hex
