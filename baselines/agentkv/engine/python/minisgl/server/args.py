from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


def _parse_cuda_graph_bs(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise argparse.ArgumentTypeError("--cuda-graph-bs values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--cuda-graph-bs must contain at least one value")
    return sorted(set(values))


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    preserve_assistant_think: bool = True
    debug_rendered_prompt: bool = False
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from minisgl.attention import validate_attn_backend
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )
    parser.add_argument(
        "--cuda-graph-bs",
        type=_parse_cuda_graph_bs,
        default=ServerArgs.cuda_graph_bs,
        help=(
            "Comma-separated explicit CUDA graph batch sizes to capture. "
            "Overrides --cuda-graph-max-bs when set."
        ),
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--preserve-assistant-think",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.preserve_assistant_think,
        help="Use a Qwen3-style chat template that preserves historical assistant <think> blocks.",
    )

    parser.add_argument(
        "--debug-rendered-prompt",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.debug_rendered_prompt,
        help="Log the final rendered prompt in the tokenizer process before tokenization.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=ServerArgs.prefill_chunk_size,
        help=(
            "Optional per-request prefill chunk cap. 0 means each request can use the "
            "full max-prefill-length budget."
        ),
    )

    parser.add_argument(
        "--eviction-plan",
        type=str,
        default=ServerArgs.eviction_plan,
        help=(
            "Session KV eviction plan preset. Supported: none, identity, "
            "streamingllm, streamingllm-512, streamingllm-1024, "
            "snapkv, snapkv-1024, snapkv-2048, snapkv-4096, snapkv-6114, "
            "snapkv-8192, snapkv-<budget>-obs<window>, or "
            "snapkv-<budget>-obs<window>-pool<kernel>; r1kv supports "
            "r1kv-<budget>-obs<window>-pool<kernel>. "
            "none bypasses eviction/compaction logic entirely. Trigger timing "
            "is controlled by compact-* flags."
        ),
    )

    parser.add_argument(
        "--compact-interval",
        type=int,
        default=ServerArgs.compact_interval,
        help="Compact every N generated tokens during decode. 0 disables interval compaction.",
    )

    parser.add_argument(
        "--compact-after-prefill",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.compact_after_prefill,
        help="Run compact after the first generated token, i.e. after prefill is materialized.",
    )

    parser.add_argument(
        "--compact-before-park",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.compact_before_park,
        help="Run compact before parking a finished session request.",
    )

    parser.add_argument(
        "--compact-aware-kv-reserve",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.compact_aware_kv_reserve,
        help=(
            "Reserve decode KV only until the next compact point when eviction "
            "compaction is active."
        ),
    )

    parser.add_argument(
        "--compact-aware-prefill-reserve",
        action=argparse.BooleanOptionalAction,
        default=ServerArgs.compact_aware_prefill_reserve,
        help=(
            "Reserve only the current prefill chunk for prompt admission when "
            "compact-aware KV reserve is enabled."
        ),
    )

    parser.add_argument(
        "--compact-debug",
        action="store_true",
        default=ServerArgs.compact_debug,
        help="Write compact debug NPZ artifacts. Disabled by default for benchmarks.",
    )

    parser.add_argument(
        "--compact-debug-dir",
        type=str,
        default=ServerArgs.compact_debug_dir,
        help="Directory for compact debug JSONL/NPZ artifacts.",
    )

    parser.add_argument(
        "--compact-debug-jsonl",
        action="store_true",
        default=ServerArgs.compact_debug_jsonl,
        help="Write compact debug JSONL summaries. Disabled by default for benchmarks.",
    )

    parser.add_argument(
        "--compact-debug-max-items",
        type=int,
        default=ServerArgs.compact_debug_max_items,
        help="Reserved sampling limit for compact debug artifacts; 0 means full artifacts.",
    )

    parser.add_argument(
        "--eviction-trace",
        action="store_true",
        default=ServerArgs.eviction_trace,
        help=(
            "Write lightweight eviction trace JSONL plus NPZ keep/drop logical-position "
            "artifacts. Disabled by default for benchmarks."
        ),
    )

    parser.add_argument(
        "--eviction-trace-dir",
        type=str,
        default=ServerArgs.eviction_trace_dir,
        help="Directory for eviction trace JSONL/NPZ artifacts.",
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["cuda_graph_bs"] = [1]
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
