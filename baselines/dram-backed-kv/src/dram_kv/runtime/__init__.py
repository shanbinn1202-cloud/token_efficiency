"""Framework-neutral DRAM-backed KV runtime."""

from .core import DramKVRuntime, OperationMetrics, RestoreResult, RuntimeStats
from .qwen import (
    ActiveQwenKV,
    FullAttentionKV,
    QwenKVPage,
    QwenKVPageAdapter,
    assemble_pages,
    extract_page,
    extract_suffix_page,
    full_attention_layers,
    write_active_cache,
)
from .qwen_forward import (
    PreparedQwenPrefill,
    QwenForwardBridge,
    QwenPreparationReceipt,
)
from .tensor import DuckTensorAdapter, TensorAdapter

__all__ = [
    "ActiveQwenKV",
    "DramKVRuntime",
    "DuckTensorAdapter",
    "FullAttentionKV",
    "OperationMetrics",
    "QwenKVPage",
    "QwenKVPageAdapter",
    "PreparedQwenPrefill",
    "QwenForwardBridge",
    "QwenPreparationReceipt",
    "RestoreResult",
    "RuntimeStats",
    "TensorAdapter",
    "assemble_pages",
    "extract_page",
    "extract_suffix_page",
    "full_attention_layers",
    "write_active_cache",
]
