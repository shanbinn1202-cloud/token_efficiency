from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    CloseSessionMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, ConfigDict, Field

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    session_id: str | None = None
    close_session: bool = False
    ignore_eos: bool = False


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    tool_calls: List[Dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None
    tools: List[Dict[str, Any]] | None = None
    tool_choice: Any = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False
    session_id: str | None = None
    close_session: bool = False


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _message_to_chat_template_dict(message: Message) -> Dict[str, Any]:
    return message.model_dump(exclude_none=True)


def _parse_qwen3_tool_calls(text: str, uid: int) -> tuple[str | None, List[Dict[str, Any]]]:
    tool_calls: List[Dict[str, Any]] = []

    for idx, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        raw_call = match.group(1).strip()
        try:
            parsed = json.loads(raw_call)
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool_call JSON for user %s: %s", uid, raw_call)
            continue
        if not isinstance(parsed, dict):
            logger.warning("Ignoring non-object tool_call for user %s: %s", uid, raw_call)
            continue

        function = parsed.get("function", parsed)
        if not isinstance(function, dict):
            logger.warning("Ignoring tool_call with non-object function for user %s: %s", uid, raw_call)
            continue
        name = function.get("name") or function.get("tool")
        if not name:
            logger.warning("Ignoring tool_call without function name for user %s: %s", uid, raw_call)
            continue
        arguments = function.get("arguments", function.get("parameters", {}))
        if not isinstance(arguments, str):
            arguments = _json_dumps(arguments)

        tool_calls.append(
            {
                "id": parsed.get("id") or f"call_{uid}_{idx}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )

    if not tool_calls:
        return text, tool_calls

    content = _TOOL_CALL_RE.sub("", text).strip()
    if tool_calls and not content:
        content = None
    return content, tool_calls


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        try:
            while True:
                await event.wait()
                event.clear()

                pending = self.ack_map[uid]
                self.ack_map[uid] = []
                ack = None
                for ack in pending:
                    if ack.error:
                        raise HTTPException(status_code=ack.status_code or 500, detail=ack.error)
                    yield ack
                if ack and ack.finished:
                    break
        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def collect_output(self, uid: int) -> str:
        output = []
        async for ack in self.wait_for_ack(uid):
            if ack.incremental_output:
                output.append(ack.incremental_output)
            if ack.finished:
                break
        logger.debug("Finished non-streaming response for user %s", uid)
        return "".join(output)

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    uid = state.new_user()
    if req.close_session:
        await state.send_one(CloseSessionMsg(uid=uid, session_id=req.session_id))
        await state.collect_output(uid)
        return {"session_id": req.session_id, "closed": True}
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            session_id=req.session_id,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )
    output = await state.collect_output(uid)
    return {"text": output, "session_id": req.session_id}


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest):
    state = get_global_state()
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if req.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported")
    if req.close_session:
        uid = state.new_user()
        await state.send_one(CloseSessionMsg(uid=uid, session_id=req.session_id))
        await state.collect_output(uid)
        return {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "session_id": req.session_id,
            "closed": True,
        }
    if req.messages:
        prompt = [_message_to_chat_template_dict(msg) for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            tools=req.tools,
            session_id=req.session_id,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    output = await state.collect_output(uid)
    content, tool_calls = _parse_qwen3_tool_calls(output, uid)
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "raw_content": output,
    }
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest) -> str:
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [_message_to_chat_template_dict(msg) for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            tools=req.tools,
            session_id=req.session_id or "shell",
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )
    return await state.collect_output(uid)


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                session_id="shell",
            )
            cur_msg = await shell_completion(req)
            print(cur_msg, flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
