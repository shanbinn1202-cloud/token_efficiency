from intentkv.intentkv import IntentKVCache
from intentkv.attn_forward import attn_forward
from typing import Dict
import functools
from transformers.cache_utils import Cache

def prepare_inputs_for_generation_intentkv(
    config, original_prepare_inputs_for_generation
):
    @functools.wraps(original_prepare_inputs_for_generation)
    def new_prepare_inputs_for_generation(self, *args, **kwargs):
        outputs = original_prepare_inputs_for_generation(*args, **kwargs)
        use_cache = kwargs.get("use_cache", True)
        if use_cache and not isinstance(
            outputs.get("past_key_values", None), IntentKVCache
        ):
            outputs["past_key_values"] = IntentKVCache(config)
        if self._supports_num_logits_to_keep():
            outputs["num_logits_to_keep"] = 1
        return outputs

    return new_prepare_inputs_for_generation

def prepare_cache(config):
    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs: Dict, *args, **kwargs
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s paramaterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """
        model_kwargs["past_key_values"] = IntentKVCache(config)

    return _prepare_cache_for_generation


def patch_model(model, config):
    print("IntentKV Config: ", config)
    Attention = model.model.layers[0].self_attn.__class__
    def update_module(m):
        if isinstance(m, Attention):
            m.forward = (
                lambda self, *args, **kwargs: attn_forward(self, *args, **kwargs)
            ).__get__(m, Attention)

    model.apply(update_module)
    prepare_cache_func = prepare_cache(config)
    model._prepare_cache_for_generation = prepare_cache_func.__get__(
        model, model.__class__
    )

    prepare_inputs_func = prepare_inputs_for_generation_intentkv(
        config, model.prepare_inputs_for_generation
    )
    model.prepare_inputs_for_generation = prepare_inputs_func.__get__(
        model, model.__class__
    )

    return model