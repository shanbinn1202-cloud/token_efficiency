"""HF model execution with the original MiniSGL paged cache and session lifecycle."""
from dataclasses import replace
import time

import torch
from tebench.backends.base import normalize_messages
from tebench.backends.execution import configure_linear, sdpa_context
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .cache import PagedSession
from .upstream import factory, phase_ids, advance_phase, semantic_phase_ids, advance_semantic_phase


class Runtime:
    def __init__(self, model_path, budget=16384, *, max_window_size=131072, prefill_chunk_size=None,
                 eviction_interval=128, cuda_graph=False, enable_thinking=True, local_files_only=True,
                 linear_attention="native", sdpa_backend="auto", policy_options=None):
        if cuda_graph:
            raise ValueError("AgentKV upstream paged cache currently requires cuda_graph=false")
        configure_linear(linear_attention)
        self.sdpa_backend = sdpa_backend
        self.max_window_size = max_window_size
        self.prefill_chunk_size = prefill_chunk_size
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        self.plan = replace(factory(f"agentkv-{budget}"), **(policy_options or {}))
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda", local_files_only=local_files_only,
            attn_implementation="sdpa",
        ).eval()
        self.storage = PagedSession(self.model.config, self.plan,
            max_window_size=max_window_size, eviction_interval=eviction_interval)
        ALL_ATTENTION_FUNCTIONS.register("agentkv_external", self.attention)
        self.model.config._attn_implementation = "agentkv_external"
        for layer in self.model.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.config._attn_implementation = "agentkv_external"
        self.reset()

    @property
    def cache(self):
        return self.storage.cache

    def reset(self):
        self.storage.reset()

    def attention(self, module, query, key, value, attention_mask, scaling=None, **kwargs):
        self.storage.query_ring.write_layer(layer_id=self.storage.layer_map[module.layer_idx],
            q=query[0].transpose(0, 1), metadata=self.storage.query_metadata)
        qlen, klen = query.shape[-2], key.shape[-2]
        # All retained keys precede the current suffix in logical time, even
        # when different heads retained different original positions.
        mask = None
        if qlen > 1 and qlen != klen:
            mask = torch.arange(klen, device=query.device)[None, :] <= (
                torch.arange(qlen, device=query.device)[:, None] + klen - qlen)
        with sdpa_context(self.sdpa_backend):
            return sdpa_attention_forward(module, query, key, value,
                attention_mask=mask, scaling=scaling, is_causal=qlen == klen)

    @torch.inference_mode()
    def forward(self):
        req = self.storage.req
        positions = self.storage.prepare_forward()
        return self.model(
            input_ids=req.input_ids[req.physical_cached_len:req.device_len].to("cuda", dtype=torch.long)[None],
            position_ids=positions.long()[None],
            attention_mask={"full_attention": None, "linear_attention": None},
            past_key_values=self.cache, use_cache=True, logits_to_keep=1,
        ).logits[0, -1]

    @torch.inference_mode()
    def generate(self, messages, tools, max_tokens=4096, temperature=0.0):
        prompt = self.tokenizer.apply_chat_template(normalize_messages(messages), tools=tools or None,
            tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        return self.generate_prompt(prompt, max_tokens=max_tokens, temperature=temperature)

    @torch.inference_mode()
    def generate_prompt(self, prompt, *, max_tokens=4096, temperature=0.0):
        """Append-only raw-text session API, also used for integration validation."""
        start = time.perf_counter()
        encoded = self.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        ids = encoded["input_ids"]
        stages = phase_ids(prompt, torch.tensor(encoded["offset_mapping"])).tolist()
        semantic_stages = semantic_phase_ids(prompt, torch.tensor(encoded["offset_mapping"])).tolist()
        if max_tokens < 1 or len(ids) + max_tokens > self.max_window_size:
            raise ValueError(f"ContextWindowExceeded: {len(ids)} + {max_tokens} > {self.max_window_size}")
        previous_compactions = self.storage.compactions
        req = self.storage.begin(ids, stages, prompt, max_tokens, temperature,
            encode_suffix=lambda text: torch.tensor(self.tokenizer.encode(text,
                add_special_tokens=False), dtype=torch.int32), semantic_stages=semantic_stages)
        # Text-prefix fallback can preserve a different tokenization from the
        # client. Check and report the authoritative server history as upstream.
        if req.session_prompt_len + max_tokens > self.max_window_size:
            raise ValueError("ContextWindowExceeded after session tokenization")
        reused = req.logical_cached_len
        physical_reused = req.physical_cached_len
        prompt_end = req.device_len
        logical_end = req.logical_device_len
        chunk_size = self.prefill_chunk_size or max(1, req.extend_len)
        while req.physical_cached_len < prompt_end:
            req.device_len = min(prompt_end, req.physical_cached_len + chunk_size)
            req.logical_device_len = req.logical_cached_len + req.extend_len
            logits = self.forward()
            if req.device_len < prompt_end:
                req.complete_prefill_chunk()
            else:
                assert req.logical_device_len == logical_end
                break
        generated = []
        eos_ids = self.model.generation_config.eos_token_id
        eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        finished_by_eos = False
        for step in range(max_tokens):
            token = int((torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), 1)
                         if temperature > 0 else logits.argmax()).item())
            # Same ordering as upstream engine + scheduler: complete device
            # work, append sampled token, compact, then park if finished.
            req.complete_one()
            tail = (req.phase_text_tail + self.tokenizer.decode([token], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False))[-128:]
            stage = advance_phase(tail, int(req.phase_ids[-1]))
            semantic_stage = advance_semantic_phase(tail, int(req.semantic_phase_ids[-1]))
            req.append_host(torch.tensor([token], dtype=torch.int32), phase_id=stage,
                semantic_phase_id=semantic_stage, phase_text_tail=tail)
            generated.append(token)
            finished_by_eos = token in eos
            finished = finished_by_eos or not req.can_decode
            self.storage.compact(finished=finished)
            if finished:
                break
            logits = self.forward()
        text = self.tokenizer.decode(generated, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        parked_text = prompt + self.tokenizer.decode(generated[:-1] if finished_by_eos else generated,
            skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.storage.park(drop_last_token=finished_by_eos, full_text=parked_text)
        torch.cuda.synchronize()
        return text, {
            "prompt_tokens": req.session_prompt_len, "completion_tokens": len(generated),
            "cached_prompt_tokens": reused, "physical_cached_prompt_tokens": physical_reused,
            "seconds": time.perf_counter() - start,
            "compactions": self.storage.compactions - previous_compactions,
            "peak_physical_kv_tokens": self.storage.peak_physical,
            "physical_kv_tokens": req.physical_cached_len,
            "logical_cached_tokens": req.logical_cached_len,
            "session_id": self.storage.session_id, "cache_rebuilds": 0,
            "cache_storage": "minisgl.MHAKVCache", "compression": "minisgl.EvictionManager/triton",
        }
