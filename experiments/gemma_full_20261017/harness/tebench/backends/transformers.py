"""Non-vLLM backend for history-selection methods; no KV selection policy."""
import time
from .base import Backend, response
from .execution import configure_linear, describe, sdpa_context


class TransformersBackend(Backend):
    def __init__(self, config):
        super().__init__(config)
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM
        self.torch = torch
        settings = config["model"]
        path, local = settings["path"], settings["local_files_only"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local)
        model_config = AutoConfig.from_pretrained(path, local_files_only=local)
        is_qwen = getattr(model_config, "model_type", "") in {"qwen3_5", "qwen3_5_text"}
        if is_qwen:
            configure_linear(config["backend"]["linear_attention"])
        cls = Qwen3_5ForCausalLM if is_qwen else AutoModelForCausalLM
        self.model = cls.from_pretrained(path, dtype=getattr(torch, settings["dtype"]),
            device_map="cuda", local_files_only=local, attn_implementation=config["backend"]["full_attention"]).eval()
        torch.manual_seed(config["generation"]["seed"])
        self.prefixes = {}
        self.execution_info = describe(self.model, config)

    def reset(self):
        super().reset()
        self.prefixes = {}

    def complete(self, *, model, messages, tools=None, **options):
        torch = self.torch
        started = time.perf_counter()
        _, ids, max_tokens = self.prepare(messages, tools, options)
        temperature = options.get("temperature", self.config["generation"]["temperature"])
        choices = (options.get("extra_body") or {}).get("structured_outputs", {}).get("choice")
        generation = self.config["generation"]
        kwargs = {"do_sample": temperature > 0, "repetition_penalty": generation["repetition_penalty"]}
        if options.get("stop"):
            kwargs.update(stop_strings=options["stop"], tokenizer=self.tokenizer)
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=generation["top_p"], top_k=max(0, generation["top_k"]))
        if choices:
            sequences = [self.tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
            eos = self.tokenizer.eos_token_id
            def allowed(batch, tokens):
                suffix = tokens[len(ids):].tolist()
                return sorted({seq[len(suffix)] if len(seq) > len(suffix) else eos
                    for seq in sequences if seq[:len(suffix)] == suffix})
            kwargs["prefix_allowed_tokens_fn"] = allowed
        kind = "agent" if tools else "auxiliary"
        reused = 0
        if self.config["backend"]["prefix_caching"] and kind in self.prefixes:
            previous, cache = self.prefixes[kind]
            if previous and len(previous) < len(ids) and ids[:len(previous)] == previous:
                kwargs["past_key_values"] = cache
                reused = len(previous)
        try:
            with torch.inference_mode(), sdpa_context(self.config["backend"]["sdpa_kernel"]):
                result = self.model.generate(input_ids=torch.tensor([ids], device=self.model.device),
                    attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=self.model.device),
                    max_new_tokens=max_tokens, use_cache=True, return_dict_in_generate=True, disable_compile=True, **kwargs)
                output = result.sequences[0, len(ids):]
        except Exception:
            self.prefixes.pop(kind, None)
            raise
        if self.config["backend"]["prefix_caching"]:
            self.prefixes[kind] = (result.sequences[0, :-1].tolist(), result.past_key_values)
        torch.cuda.synchronize()
        text = self.tokenizer.decode(output, skip_special_tokens=False)
        if options.get("stop"):
            stops = options["stop"]
            if isinstance(stops, str):
                stops = [stops]
            text = text[:min((text.index(s) for s in stops if s in text), default=len(text))]
        self.last_measurement = {"seconds": time.perf_counter() - started, "physical_kv_policy": "token_window_reprefill" if "_prompt_token_ids" in options else "full",
                                 "prompt_tokens": len(ids), "completion_tokens": len(output), "cached_prompt_tokens": reused}
        return response(model, text, len(ids), len(output), tools,
            cached_tokens=reused, finish_reason="length" if len(output) == max_tokens else "stop",
            text_only=options.get("_text_response", False),
            enable_thinking=(options.get("extra_body") or {}).get("chat_template_kwargs", {}).get(
                "enable_thinking", self.config["model"]["enable_thinking"]))

    def close(self):
        self.reset()
        del self.model
        self.torch.cuda.empty_cache()
