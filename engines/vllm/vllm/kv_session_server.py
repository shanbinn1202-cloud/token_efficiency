"""Local-only HTTP bridge to vLLM's native resumable streaming requests."""
import asyncio
from pathlib import Path
import uuid

from starlette.responses import JSONResponse
from vllm import SamplingParams
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind
from vllm.kv_window import select_window


class Session:
    def __init__(self, engine, method="sliding_window"):
        self.method = method
        self.prompt_text = ""
        self.pending_token = None
        self.engine = engine
        self.rid = "kv-window-" + uuid.uuid4().hex
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.task = None
        self.future = None
        self.logical = self.computed = self.removed = 0
        self.starts = []
        self.prefix = None
        self.error = None
        self.dram_controller = None
        self.repair_controller = None
        self.agent_history = []
        self.agent_boundaries = []

    async def inputs(self):
        while (item := await self.queue.get()) is not None:
            yield item

    async def consume(self, params):
        ids = []
        text = ""
        try:
            async for output in self.engine.generate(self.inputs(), params, self.rid):
                for part in output.outputs:
                    ids.extend(part.token_ids)
                    text += part.text
                    if part.finish_reason is not None and self.future and not self.future.done():
                        self.future.set_result((ids, text, part.finish_reason))
                        ids, text = [], ""
        except Exception as error:
            self.error = error
            if self.future and not self.future.done():
                self.future.set_exception(error)

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.dram_controller is not None:
            await self.dram_controller.close()
        if self.repair_controller is not None:
            await self.repair_controller.close()
        if self.method == 'agentkv':
            await self.engine.collective_rpc('agentkv_release', args=(self.rid,))

    async def complete(self, body):
        if self.method == "repairkv":
            async with self.lock:
                if self.error:
                    raise RuntimeError(f"Session failed: {self.error}")
                if self.repair_controller is None:
                    from adapters.repairkv.session import Controller
                    self.repair_controller = Controller(self, body)
                return await self.repair_controller.complete(body)
        if self.method == "agentkv":
            return await self.complete_agentkv(body)
        if self.method == "dram_backed_kv":
            return await self.complete_dram(body)
        async with self.lock:
            if self.error:
                raise RuntimeError(f"Session failed: {self.error}")
            ids = body["input_ids"]
            available_input = self.computed + len(ids)
            original_incoming = len(ids)
            header = body["header_tokens"]
            budget = body["budget"]
            if not ids or not 0 < header <= len(ids):
                raise ValueError("Invalid input/header")
            if self.prefix is None:
                self.prefix = body["prefix_tokens"]
                if self.prefix != len(ids) - header or len(ids) > budget:
                    raise ValueError("Initial prefix/header exceeds budget or is inconsistent")
                kept, start, incoming = 0, 0, len(ids)
                retained = 0
                meta = dict(removed=0)
                next_starts = [self.prefix]
            else:
                start, kept, incoming = select_window(self.prefix, self.starts, self.computed,
                                                     len(ids), header, budget)
                ids = ids[-incoming:]
                retained = self.prefix + self.computed - start
                meta = dict(prefix=self.prefix, suffix_start=start, computed=self.computed,
                            logical_computed=self.logical, budget=budget,
                            removed=self.removed + start - self.prefix)
                next_starts = [self.prefix + s - start for s in self.starts if s >= start]
                next_starts.append(retained + len(ids) - header)
            if self.logical + len(ids) + body["max_tokens"] > self.engine.model_config.max_model_len:
                raise ValueError("Logical RoPE context limit reached")
            params = SamplingParams(max_tokens=body["max_tokens"], temperature=body["temperature"],
                top_p=body.get("top_p", 1.0), top_k=body.get("top_k", -1),
                repetition_penalty=body.get("repetition_penalty", 1.0), seed=body.get("seed", 0),
                skip_special_tokens=False, output_kind=RequestOutputKind.DELTA,
                extra_args={"kv_window": meta})
            self.future = asyncio.get_running_loop().create_future()
            await self.queue.put(StreamingInput(prompt={"prompt_token_ids": ids}, sampling_params=params))
            if self.task is None:
                self.task = asyncio.create_task(self.consume(params))
            outputs, text, reason = await self.future
            # Native streaming retains all but the final sampled (uncomputed) token.
            old_removed = self.removed
            previous_logical = self.logical
            self.logical += len(ids) + len(outputs) - 1
            self.computed = retained + len(ids) + len(outputs) - 1
            self.removed = self.logical - self.computed
            self.starts = next_starts
            return dict(text=text, output_ids=outputs, finish_reason=reason,
                        prompt_tokens=retained + len(ids), completion_tokens=len(outputs),
                        cached_tokens=retained, prefill_tokens=len(ids), retained_turns=kept,
                        evicted_tokens=meta.get("removed", 0) - old_removed,
                        logical_computed_tokens=self.logical, retained_kv_tokens=self.computed,
                        total_evicted_tokens=self.removed, selection="recent_tokens",
                        available_input_tokens=available_input,
                        dropped_input_tokens=original_incoming - len(ids))

    async def complete_dram(self, body):
        async with self.lock:
            if self.error:
                raise RuntimeError(f"Session failed: {self.error}")
            if self.dram_controller is None:
                from adapters.dram_backed_kv.vllm_session import Controller
                self.dram_controller = Controller(self, body)
            return await self.dram_controller.complete(body)

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
            max_tokens = output_limit(self.logical + len(ids), body["max_tokens"],
                                      self.engine.model_config.max_model_len)
            tokenizer = self.engine.get_tokenizer()
            # A non-EOS final sample has no KV yet. The client restores its raw
            # ID; it belongs to the previous generation, not the encoded suffix.
            # Byte tokens need not survive decode/encode in isolation.
            restored = int(self.pending_token is not None)
            if restored and ids[0] != self.pending_token:
                raise ValueError("Restored token differs from the parked final sample")
            suffix_ids = ids[restored:]
            fresh = tokenizer.decode(suffix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            encoded = tokenizer(fresh, add_special_tokens=False, return_offsets_mapping=True)
            if encoded["input_ids"] != suffix_ids:
                raise ValueError("Non-canonical suffix tokenization; cannot assign exact upstream phases")
            prompt = self.prompt_text + fresh
            offsets = torch.tensor(encoded["offset_mapping"]) + len(self.prompt_text)
            stages = phase_ids(prompt, offsets).tolist() if suffix_ids else []
            # The worker restores the sampled token's rolling generation phase
            # from its parked state, before adopting the new prompt tail.
            meta = dict(phase_ids=[-1] * restored + stages, restored_tokens=restored,
                        history_start=self.logical,
                        phase_tail=prompt[-128:], session_id=self.rid,
                        logical=self.logical, physical=self.computed,
                        boundaries=self.agent_boundaries)
            params = SamplingParams(max_tokens=max_tokens, temperature=body["temperature"],
                ignore_eos=bool(body.get("ignore_eos", False)),
                top_p=body.get("top_p", 1.0), top_k=body.get("top_k", -1),
                repetition_penalty=body.get("repetition_penalty", 1.0), seed=body.get("seed", 0),
                skip_special_tokens=False, output_kind=RequestOutputKind.DELTA,
                extra_args={"agentkv": meta})
            outputs, text, reason = [], '', None
            request_id = self.rid + '-' + uuid.uuid4().hex
            async for result in self.engine.generate(
                    {'prompt_token_ids': self.agent_history + ids}, params, request_id):
                for part in result.outputs:
                    outputs.extend(part.token_ids)
                    text += part.text
                    reason = part.finish_reason or reason
            self.logical += len(ids) + len(outputs) - 1
            self.agent_history.extend(ids + outputs[:-1])
            assert len(self.agent_history) == self.logical
            eos = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
            self.pending_token = outputs[-1] if outputs and outputs[-1] not in eos else None
            # Match upstream _session_full_text: drop only EOS, not a length-
            # limited final sample. Decode the whole generated span together.
            kept_outputs = outputs if self.pending_token is not None else outputs[:-1]
            self.prompt_text = prompt + tokenizer.decode(kept_outputs, skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            stats, = await self.engine.collective_rpc("kv_session_stats", args=(self.rid,))
            assert stats["logical_computed_tokens"] == self.logical
            self.computed = stats['retained_kv_tokens']
            self.agent_boundaries = stats.pop('boundaries')
            return dict(text=text, output_ids=outputs, finish_reason=reason,
                completion_tokens=len(outputs), **stats)


async def middleware(request, call_next):
    if not any(request.url.path.startswith(p) for p in ("/v1/kv-window/", "/v1/kv-session/")):
        return await call_next(request)
    import os
    import json
    method = ("repairkv" if os.environ.get("VLLM_REPAIRKV") == "1" else
              "agentkv" if os.environ.get("VLLM_AGENTKV") == "1" else
              "dram_backed_kv" if os.environ.get("VLLM_DRAM_KV") == "1" else
              "sliding_window")
    state = request.app.state
    engine = state.engine_client
    if not hasattr(state, "kv_window_sessions"):
        state.kv_window_sessions = {}
    sessions = state.kv_window_sessions
    if request.url.path.endswith("/info"):
        env_prefix = {"repairkv": "VLLM_REPAIRKV", "dram_backed_kv": "VLLM_DRAM_KV"}.get(method, "VLLM_AGENTKV")
        return JSONResponse(dict(engine_path=str(Path(__file__).resolve().parent),
                                 implementation="native-request-kv-policy-v3", method=method, sessions=len(sessions),
                                 async_scheduling=engine.vllm_config.scheduler_config.async_scheduling,
                                 scheduler=engine.vllm_config.scheduler_config.get_scheduler_cls().__name__,
                                 scheduling_policy=str(engine.vllm_config.scheduler_config.policy),
                                 budget=int(os.environ.get(env_prefix + "_BUDGET", "0")),
                                 options=json.loads(os.environ.get(env_prefix + "_OPTIONS", "{}"))))
    try:
        config = engine.vllm_config
        if config.cache_config.enable_prefix_caching:
            raise ValueError("In-place KV policies require --no-enable-prefix-caching")
        if config.scheduler_config.async_scheduling and method == "dram_backed_kv":
            raise ValueError("DRAM streaming has not been adapted to asynchronous execution")
        if config.parallel_config.tensor_parallel_size != 1 or config.speculative_config is not None:
            raise ValueError("KV window supports TP=1 without speculative decoding")
        body = await request.json()
        sid = body["session_id"]
        if request.url.path.endswith("/close"):
            session = sessions.pop(sid, None)
            if session:
                await session.close()
            return JSONResponse({"closed": True})
        if body.get("method", "sliding_window") != method:
            raise ValueError("Client policy differs from running server")
        if method == "agentkv" and body["budget"] != int(os.environ["VLLM_AGENTKV_BUDGET"]):
            raise ValueError("Client budget differs from running AgentKV server")
        if method == "dram_backed_kv" and body["budget"] != int(os.environ["VLLM_DRAM_KV_BUDGET"]):
            raise ValueError("Client budget differs from running DRAM server")
        if method == "repairkv" and body["budget"] != int(os.environ["VLLM_REPAIRKV_BUDGET"]):
            raise ValueError("Client budget differs from running RepairKV server")
        session = sessions.setdefault(sid, Session(engine, method))
        return JSONResponse(await session.complete(body))
    except Exception as error:
        return JSONResponse({"error": str(error), "type": type(error).__name__}, status_code=400)
