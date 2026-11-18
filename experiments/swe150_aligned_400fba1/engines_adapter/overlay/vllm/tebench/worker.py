"""Method RPCs exposed through worker_extension_cls; native Worker is unchanged."""

import torch


class WorkerExtension:
    def runtime_backend_info(self):
        import vllm

        runner = self.model_runner
        layers = runner.compilation_config.static_forward_context
        attention = {
            name: type(layer.impl).__name__
            for name, layer in layers.items()
            if getattr(layer, "impl", None) is not None
        }
        gdn = {
            name: layer.gdn_prefill_backend
            for name, layer in runner.model.named_modules()
            if hasattr(layer, "gdn_prefill_backend")
        }
        return dict(version=vllm.__version__, source=vllm.__file__,
                    attention=attention, gdn=gdn,
                    async_scheduling=runner.use_async_scheduling,
                    max_model_len=runner.max_model_len,
                    max_num_batched_tokens=runner.scheduler_config.max_num_batched_tokens)

    def kv_session_recompute_stats(self):
        from vllm.tebench.kv_recompute import CLOSED_TOTALS, HISTORIES, stats

        sessions = {rid: stats(rid) for rid in HISTORIES}
        totals = dict(CLOSED_TOTALS)
        for key in (
            "preemptions",
            "running_preemptions",
            "recomputed_tokens",
            "replayed_compactions",
        ):
            totals[key] = totals.get(key, 0) + sum(s[key] for s in sessions.values())
        config = self.model_runner.kv_cache_config
        return dict(
            totals=totals,
            sessions=sessions,
            kv_pool_bytes=sum(t.size for t in config.kv_cache_tensors),
            kv_pool_blocks=config.num_blocks,
            groups=[
                dict(
                    layers=len(g.layer_names),
                    type=type(g.kv_cache_spec).__name__,
                    block_size=g.kv_cache_spec.block_size,
                    page_size_bytes=g.kv_cache_spec.page_size_bytes,
                )
                for g in config.kv_cache_groups
            ],
            cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_reserved_bytes=torch.cuda.memory_reserved(),
        )

    def kv_session_stats(self, rid, seal=True):
        import os

        if os.environ.get("VLLM_INTENTKV") == "1":
            from vllm.tebench.intentkv import get_stats

            return get_stats(rid)
        if (
            os.environ.get("VLLM_RANDOM_ATTENTION") == "1"
            or os.environ.get("VLLM_COMPACT_WINDOW") == "1"
        ):
            from vllm.tebench.random_attention import get_stats

            return get_stats(rid)
        if os.environ.get("VLLM_NATIVE_WINDOW") == "1":
            from vllm.tebench.native_window import get_stats

            return get_stats(rid)
        if os.environ.get("VLLM_REPAIRKV") == "1":
            from vllm.tebench.repairkv import get_stats

            return get_stats(self.model_runner, rid)
        from vllm.tebench.agentkv import get_stats

        return get_stats(self, rid)

    def repairkv_prepare(self, rid, answer_tokens):
        from vllm.tebench.repairkv import prepare

        return prepare(self.model_runner, rid, answer_tokens)

    def repairkv_select(self, rid):
        from vllm.tebench.repairkv import select

        return select(self.model_runner, rid)

    def repairkv_release(self, rid):
        from vllm.tebench.repairkv import release

        release(rid)

    def agentkv_release(self, rid):
        from vllm.tebench.agentkv import release

        release(rid)

    def native_window_release(self, rid):
        from vllm.tebench.native_window import EXTERNAL

        EXTERNAL.pop(rid, None)

    def random_attention_release(self, rid):
        from vllm.tebench.random_attention import EXTERNAL

        EXTERNAL.pop(rid, None)

    def intentkv_release(self, rid):
        from vllm.tebench.intentkv import EXTERNAL

        EXTERNAL.pop(rid, None)
