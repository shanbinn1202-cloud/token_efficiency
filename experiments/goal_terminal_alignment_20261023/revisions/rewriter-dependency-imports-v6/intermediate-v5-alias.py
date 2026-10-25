"""Compatibility export for lm-format-enforcer 0.11.x on vLLM 0.24.

The tokenizer implementation is unchanged; vLLM moved this class to tokenizers.
"""
from vllm.tokenizers.mistral import MistralTokenizer

__all__ = ["MistralTokenizer"]
