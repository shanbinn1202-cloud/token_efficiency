from __future__ import annotations

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    CloseSessionBackendMsg,
    CloseSessionMsg,
    DetokenizeMsg,
    TokenizeMsg,
    TokenizerErrorMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


QWEN3_PRESERVE_ASSISTANT_THINK_CHAT_TEMPLATE = r"""
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if message.role == 'system' and loop.first %}
    {%- elif message.role == 'user' or message.role == 'system' %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {%- if message.reasoning_content is string and '<think>' not in content %}
            {%- set content = '<think>\n' + message.reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') %}
        {%- endif %}
        {{- '<|im_start|>assistant\n' + content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == 'tool' %}
        {%- if loop.first or (messages[loop.index0 - 1].role != 'tool') %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' + content + '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
""".strip()


def _patch_chat_template_for_preserved_think(tokenizer) -> None:
    tokenizer.chat_template = QWEN3_PRESERVE_ASSISTANT_THINK_CHAT_TEMPLATE


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    preserve_assistant_think: bool = False,
    debug_rendered_prompt: bool = False,
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")
    if preserve_assistant_think:
        _patch_chat_template_for_preserved_think(tokenizer)
        logger.info("Using chat template that preserves historical assistant <think> blocks")

    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    tokenize_manager = TokenizeManager(
        tokenizer,
        debug_rendered_prompt=debug_rendered_prompt,
        logger=logger,
    )
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            close_session_msg = [m for m in pending_msg if isinstance(m, CloseSessionMsg)]
            error_msg = [m for m in pending_msg if isinstance(m, TokenizerErrorMsg)]
            assert (
                len(detokenize_msg)
                + len(tokenize_msg)
                + len(abort_msg)
                + len(close_session_msg)
                + len(error_msg)
                == len(pending_msg)
            )
            if len(detokenize_msg) > 0:
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(error_msg) > 0:
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output="",
                            finished=True,
                            error=msg.error,
                            status_code=msg.status_code,
                        )
                        for msg in error_msg
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(tokenize_msg) > 0:
                tokenized = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                            session_id=msg.session_id,
                            rendered_text=prompt,
                            phase_ids=phase_ids,
                            semantic_phase_ids=semantic_phase_ids,
                        )
                        for msg, (t, prompt, phase_ids, semantic_phase_ids) in zip(
                            tokenize_msg,
                            tokenized,
                            strict=True,
                        )
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
            if len(close_session_msg) > 0:
                batch_output = BatchBackendMsg(
                    data=[
                        CloseSessionBackendMsg(uid=msg.uid, session_id=msg.session_id)
                        for msg in close_session_msg
                    ]
                )
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        pass
