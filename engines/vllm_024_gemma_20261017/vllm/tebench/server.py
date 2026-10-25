"""Local HTTP sessions composed of ordinary native vLLM requests."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from starlette.responses import JSONResponse
from tebench.backends.model_chat import eos_ids

import vllm
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind


class Session:
    def __init__(self, engine, method="agentkv"):
        if method not in {
            "agentkv", "repairkv", "sliding_window", "random_attention", "intentkv"
        }:
            raise ValueError(f"Unsupported KV session method: {method}")
        self.engine = engine
        self.method = method
        self.rid = "kv-session-" + uuid.uuid4().hex
        self.lock = asyncio.Lock()
        self.error = None
        self.prompt_text = ""
        self.pending_token = None
        self.logical = self.computed = 0
        self.repair_controller = None
        self.agent_history = []
        self.agent_boundaries = []

    async def close(self):
        compact = (
            self.method == "sliding_window"
            and os.environ.get("VLLM_COMPACT_WINDOW") == "1"
        )
        if self.method == "random_attention" or compact:
            await self.engine.collective_rpc(
                "random_attention_release", args=(self.rid,)
            )
        if self.method == "sliding_window" and not compact:
            await self.engine.collective_rpc("native_window_release", args=(self.rid,))
        if self.repair_controller is not None:
            await self.repair_controller.close()
        if self.method == "agentkv":
            await self.engine.collective_rpc("agentkv_release", args=(self.rid,))
        if self.method == "intentkv":
            await self.engine.collective_rpc("intentkv_release", args=(self.rid,))

    async def complete(self, body):
        if self.method in {"random_attention", "intentkv"} or (
            self.method == "sliding_window"
            and os.environ.get("VLLM_COMPACT_WINDOW") == "1"
        ):
            return await self.complete_random(body)
        if self.method == "sliding_window":
            return await self.complete_window(body)
        if self.method == "agentkv":
            return await self.complete_agentkv(body)
        async with self.lock:
            if self.error:
                raise RuntimeError(f"Session failed: {self.error}")
            if self.repair_controller is None:
                from adapters.repairkv.session import Controller

                self.repair_controller = Controller(self, body)
            return await self.repair_controller.complete(body)

    async def complete_agentkv(self, body):
        async with self.lock:
            import torch
            from adapters.agentkv.vllm_policy import phase_ids

            if self.error:
                raise RuntimeError(f"Session failed: {self.error}")
            ids = body["input_ids"]
            if not ids or body["max_tokens"] < 1:
                raise ValueError("Empty input or invalid output limit")
            from adapters.agentkv.vllm_policy import output_limit

            max_tokens = output_limit(
                self.logical + len(ids),
                body["max_tokens"],
                self.engine.model_config.max_model_len,
            )
            tokenizer = self.engine.get_tokenizer()
            # A non-EOS final sample has no KV yet. The client restores its raw
            # ID; it belongs to the previous generation, not the encoded suffix.
            # Byte tokens need not survive decode/encode in isolation.
            restored = int(self.pending_token is not None)
            if restored and ids[0] != self.pending_token:
                raise ValueError("Restored token differs from the parked final sample")
            suffix_ids = ids[restored:]
            fresh = tokenizer.decode(
                suffix_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            encoded = tokenizer(
                fresh, add_special_tokens=False, return_offsets_mapping=True
            )
            if encoded["input_ids"] != suffix_ids:
                raise ValueError(
                    "Non-canonical suffix tokenization; cannot assign "
                    "exact upstream phases"
                )
            prompt = self.prompt_text + fresh
            offsets = torch.tensor(encoded["offset_mapping"]) + len(self.prompt_text)
            stages = phase_ids(prompt, offsets).tolist() if suffix_ids else []
            # The worker restores the sampled token's rolling generation phase
            # from its parked state, before adopting the new prompt tail.
            meta = dict(
                phase_ids=[-1] * restored + stages,
                restored_tokens=restored,
                history_start=self.logical,
                phase_tail=prompt[-128:],
                session_id=self.rid,
                logical=self.logical,
                physical=self.computed,
                boundaries=self.agent_boundaries,
            )
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=body["temperature"],
                ignore_eos=bool(body.get("ignore_eos", False)),
                stop_token_ids=([] if body.get("ignore_eos") else eos_ids(self.engine.get_tokenizer())),
                top_p=body.get("top_p", 1.0),
                top_k=body.get("top_k", -1),
                repetition_penalty=body.get("repetition_penalty", 1.0),
                seed=body.get("seed", 0),
                skip_special_tokens=False,
                output_kind=RequestOutputKind.DELTA,
                extra_args={"agentkv": meta},
            )
            outputs, text, reason = [], "", None
            request_id = self.rid + "-" + uuid.uuid4().hex
            async for result in self.engine.generate(
                {"prompt_token_ids": self.agent_history + ids}, params, request_id
            ):
                for part in result.outputs:
                    outputs.extend(part.token_ids)
                    text += part.text
                    reason = part.finish_reason or reason
            self.logical += len(ids) + len(outputs) - 1
            self.agent_history.extend(ids + outputs[:-1])
            assert len(self.agent_history) == self.logical
            eos = eos_ids(tokenizer)
            self.pending_token = (
                outputs[-1] if outputs and outputs[-1] not in eos else None
            )
            # Match upstream _session_full_text: drop only EOS, not a length-
            # limited final sample. Decode the whole generated span together.
            kept_outputs = outputs if self.pending_token is not None else outputs[:-1]
            self.prompt_text = prompt + tokenizer.decode(
                kept_outputs,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            (stats,) = await self.engine.collective_rpc(
                "kv_session_stats", args=(self.rid,)
            )
            assert stats["logical_computed_tokens"] == self.logical
            self.computed = stats["retained_kv_tokens"]
            self.agent_boundaries = stats.pop("boundaries")
            return dict(
                text=text,
                output_ids=outputs,
                finish_reason=reason,
                completion_tokens=len(outputs),
                **stats,
            )

    async def complete_window(self, body):
        async with self.lock:
            ids = body["input_ids"]
            prefix = body["prefix_tokens"]
            if not ids or not 0 <= prefix < body["budget"]:
                raise ValueError("Initial prefix must leave room inside the KV budget")
            if self.pending_token is not None and ids[0] != self.pending_token:
                raise ValueError("Restored token differs from the final sample")
            remaining = self.engine.model_config.max_model_len - self.logical - len(ids)
            if remaining < 1:
                raise ValueError("Logical RoPE context limit reached")
            params = SamplingParams(
                max_tokens=min(body["max_tokens"], remaining),
                temperature=body["temperature"],
                ignore_eos=bool(body.get("ignore_eos", False)),
                stop_token_ids=([] if body.get("ignore_eos") else eos_ids(self.engine.get_tokenizer())),
                top_p=body.get("top_p", 1.0),
                top_k=body.get("top_k", -1),
                repetition_penalty=body.get("repetition_penalty", 1.0),
                seed=body.get("seed", 0),
                skip_special_tokens=False,
                output_kind=RequestOutputKind.DELTA,
                extra_args={
                    "native_window": dict(
                        session_id=self.rid, prefix=prefix, logical=self.logical
                    )
                },
            )
            outputs, text, reason = [], "", None
            async for result in self.engine.generate(
                {"prompt_token_ids": self.agent_history + ids},
                params,
                self.rid + "-" + uuid.uuid4().hex,
            ):
                for part in result.outputs:
                    outputs.extend(part.token_ids)
                    text += part.text
                    reason = part.finish_reason or reason
            assert outputs
            self.agent_history.extend(ids + outputs[:-1])
            self.logical = len(self.agent_history)
            tokenizer = self.engine.get_tokenizer()
            eos = eos_ids(tokenizer)
            self.pending_token = outputs[-1] if outputs[-1] not in eos else None
            (stats,) = await self.engine.collective_rpc(
                "kv_session_stats", args=(self.rid,)
            )
            assert stats["logical_computed_tokens"] == self.logical
            assert stats["prefix_tokens"] == prefix
            return dict(
                text=text,
                output_ids=outputs,
                finish_reason=reason,
                completion_tokens=len(outputs),
                **stats,
            )

    async def complete_random(self, body):
        async with self.lock:
            compact = self.method == "sliding_window"
            intent = self.method == "intentkv"
            ids = body["input_ids"]
            prefix = body["prefix_tokens"]
            if not ids or (
                not intent and not 0 <= prefix < body["budget"] - (0 if compact else 64)
            ):
                raise ValueError("Initial prefix must leave room inside the KV budget")
            if self.pending_token is not None and ids[0] != self.pending_token:
                raise ValueError("Restored token differs from the final sample")
            remaining = self.engine.model_config.max_model_len - self.logical - len(ids)
            if remaining < 1:
                raise ValueError("Logical RoPE context limit reached")
            params = SamplingParams(
                max_tokens=min(body["max_tokens"], remaining),
                temperature=body["temperature"],
                ignore_eos=bool(body.get("ignore_eos", False)),
                stop_token_ids=([] if body.get("ignore_eos") else eos_ids(self.engine.get_tokenizer())),
                top_p=body.get("top_p", 1.0),
                top_k=body.get("top_k", -1),
                repetition_penalty=body.get("repetition_penalty", 1.0),
                seed=body.get("seed", 0),
                skip_special_tokens=False,
                output_kind=RequestOutputKind.DELTA,
                extra_args={
                    ("compact_window" if compact else self.method): dict(
                        session_id=self.rid,
                        prefix=prefix,
                        logical=self.logical,
                        physical=self.computed,
                        boundaries=self.agent_boundaries,
                    )
                },
            )
            outputs, text, reason = [], "", None
            async for result in self.engine.generate(
                {"prompt_token_ids": self.agent_history + ids},
                params,
                self.rid + "-" + uuid.uuid4().hex,
            ):
                for part in result.outputs:
                    outputs.extend(part.token_ids)
                    text += part.text
                    reason = part.finish_reason or reason
            assert outputs
            self.agent_history.extend(ids + outputs[:-1])
            self.logical = len(self.agent_history)
            tokenizer = self.engine.get_tokenizer()
            eos = eos_ids(tokenizer)
            self.pending_token = outputs[-1] if outputs[-1] not in eos else None
            (stats,) = await self.engine.collective_rpc(
                "kv_session_stats", args=(self.rid,)
            )
            assert stats["logical_computed_tokens"] == self.logical
            if not intent:
                assert stats["prefix_tokens"] == prefix
            self.computed = stats["retained_kv_tokens"]
            self.agent_boundaries = stats.pop("boundaries")
            return dict(
                text=text,
                output_ids=outputs,
                finish_reason=reason,
                completion_tokens=len(outputs),
                **stats,
            )


async def middleware(request, call_next):
    if not request.url.path.startswith("/v1/kv-session/"):
        return await call_next(request)
    method = "repairkv" if os.environ.get("VLLM_REPAIRKV") == "1" else "agentkv"
    if os.environ.get("VLLM_NATIVE_WINDOW") == "1":
        method = "sliding_window"
    if os.environ.get("VLLM_RANDOM_ATTENTION") == "1":
        method = "random_attention"
    if os.environ.get("VLLM_COMPACT_WINDOW") == "1":
        method = "sliding_window"
    if os.environ.get("VLLM_INTENTKV") == "1":
        method = "intentkv"
    prefix = (
        "VLLM_NATIVE_WINDOW" if method == "sliding_window" else "VLLM_" + method.upper()
    )
    if os.environ.get("VLLM_COMPACT_WINDOW") == "1":
        prefix = "VLLM_COMPACT_WINDOW"
    engine = request.app.state.engine_client
    if not hasattr(request.app.state, "research_kv_sessions"):
        request.app.state.research_kv_sessions = {}
    sessions = request.app.state.research_kv_sessions
    if request.url.path.endswith("/info"):
        return JSONResponse(
            dict(
                engine_path=str(Path(vllm.__file__).resolve().parent),
                implementation="native-request-kv-policy-v4-clean",
                method=method,
                sessions=len(sessions),
                async_scheduling=engine.vllm_config.scheduler_config.async_scheduling,
                scheduler=engine.vllm_config.scheduler_config.get_scheduler_cls().__name__,
                scheduling_policy=str(engine.vllm_config.scheduler_config.policy),
                budget=int(os.environ[prefix + "_BUDGET"]),
                options=json.loads(os.environ[prefix + "_OPTIONS"]),
            )
        )
    try:
        config = engine.vllm_config
        if config.cache_config.enable_prefix_caching:
            raise ValueError("In-place KV policies require disabled prefix caching")
        if (
            config.parallel_config.tensor_parallel_size != 1
            or config.speculative_config
        ):
            raise ValueError("This adapter supports TP=1 without speculative decoding")
        body = await request.json()
        sid = body["session_id"]
        if request.url.path.endswith("/close"):
            session = sessions.pop(sid, None)
            if session:
                await session.close()
            return JSONResponse({"closed": True})
        if body.get("method") != method:
            raise ValueError("Client policy differs from the server")
        if body["budget"] != int(os.environ[prefix + "_BUDGET"]):
            raise ValueError("Client budget differs from the server")
        session = sessions.setdefault(sid, Session(engine, method))
        return JSONResponse(await session.complete(body))
    except Exception as error:
        return JSONResponse(
            {"error": str(error), "type": type(error).__name__}, status_code=400
        )
